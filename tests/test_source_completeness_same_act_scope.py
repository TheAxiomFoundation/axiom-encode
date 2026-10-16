"""Scope of the session-law ("section N of this act") deferral reader.

The reader accepts a deferral that names one exact same-act section the
current branch is made subject to, because a session law or a statute that
says "section N of this act" points at its own enactment, which the corpus
does not supply as a separate provision.

In regulations, CMS demonstration terms, manuals and other non-statute
documents, "the Act" is a defined term for a separate, codified enabling
statute (the Social Security Act, the Food and Nutrition Act). A reason that
names "section 1927 of the Act" there is not a same-act dependency, so the
reader must not accept it. The negative controls below use real corpus text
(42 CFR 447.504(d)(4), 447.522(e), 423.2768(g), 435.562(e), HAR 17-1737,
a CMS 1115 STC) plus synthetic shapes; main 0.2.2203 accepts every one.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import recognize_source_structure

# Paragraphs (a)-(c) are abbreviated so that (d)(1)-(5) nest under (d).
CFR_447_504_D = (
    "(a) Definitions. For the purpose of this section, the following "
    "definitions apply.\n\n"
    "(b) Sales included in AMP. AMP includes sales to retail community "
    "pharmacies.\n\n"
    "(c) Sales excluded from AMP. AMP excludes the following:\n\n"
    "(1) Sales to clinics.\n\n"
    "(2) Sales to hospitals.\n\n"
    "(d) Sales, nominal price sales, and associated discounts, rebates, "
    "payments, or other financial transactions included in AMP for 5i drugs "
    "that are not generally dispensed through retail community pharmacies. "
    "Except for those sales, nominal price sales, and associated discounts, "
    "rebates, payments, and other financial transactions identified in "
    "paragraph (e) of this section, AMP for inhalation, infusion, instilled, "
    "implanted, or injectable drugs (5i) covered outpatient drugs identified "
    "in accordance with § 447.507 shall include sales, nominal price sales, "
    "and associated discounts, rebates, payments, or other financial "
    "transactions to all entities specified in paragraph (b) of this section, "
    "as well as the following sales, nominal price sales, and associated "
    "discounts, rebates, payments, or other financial transactions:\n\n"
    "(1) Sales to physicians.\n\n"
    "(2) Sales to pharmacy benefit managers.\n\n"
    "(3) Sales to health maintenance organizations (HMOs), including managed "
    "care organizations (MCOs).\n\n"
    "(4) Sales to insurers (except for rebates under section 1927 of the Act "
    "and this subpart).\n\n"
    "(5) Sales to hospitals.\n"
)
CFR_447_522_E = (
    "(e) Investigational drugs and other drugs are not subject to the rebate "
    "requirements of section 1927 of the Act provided they do not meet the "
    "definition of a covered outpatient drug as set forth in section 1927(k) "
    "of the Act."
)
CFR_423_2768_G = (
    "(g) Other applicable provisions. The provisions of section 1128A of the "
    "Act (except subsections (a) and (b) of section 1128A of the Act) apply to "
    "civil money penalties under this section to the same extent that they "
    "apply to a civil money penalty or procedures under section 1128A of the "
    "Act."
)
CFR_435_562_E = (
    "(e) Corrective action and additional outreach notices. The agency may be "
    "subject to corrective action under section 1904 of the Act, additional "
    "data collection, or a requirement to send additional outreach notices "
    "under Sec. 435.561(b)(3)(v), when-- (1) Reported data are not timely, "
    "complete, or of sufficient quality; or (2) Reported data or other "
    "available information indicate a failure to comply substantially with "
    "Sec. Sec. 435.550 through 435.562."
)
HAR_17_1737_DRUG_CATEGORIES = (
    "(4) The following categories of drugs, subject to restriction under "
    "section 1927 of the Act, are not covered: (A) Used for cosmetic purposes "
    "or hair growth; (B) With associated tests or monitoring purchased "
    "exclusively from the manufacturer or designee as a condition of sale; "
    "(C) Which are classed as less than effective; and (D) Used to promote "
    "fertility."
)
CMS_1115_STC_AMENDMENT_PROCESS = (
    "6. Changes Subject to the Amendment Process. Changes related to "
    "eligibility, enrollment, benefits, beneficiary rights, delivery systems, "
    "cost sharing, sources of non-federal share of funding, budget neutrality, "
    "and other comparable program elements must be submitted to CMS as "
    "amendments to the demonstration. All amendment requests are subject to "
    "approval at the discretion of the Secretary in accordance with section "
    "1115 of the Act. The state must not implement changes to these elements "
    "without prior approval by CMS."
)

# Real "the Act" shapes outside statutes: the Act is a codified enabling law.
NOT_SAME_ACT_CASES = [
    pytest.param(
        "us/regulation/42/447/504",
        CFR_447_504_D,
        ("d", "4"),
        "1927",
        "the Act",
        id="cfr-447.504(d)(4)-amp-insurer-sales",
    ),
    pytest.param(
        "us/regulation/42/447/504",
        CFR_447_504_D,
        ("d",),
        "1927",
        "the Act",
        id="cfr-447.504(d)-amp-5i-inclusions",
    ),
    pytest.param(
        "us/regulation/42/447/522",
        CFR_447_522_E,
        ("e",),
        "1927",
        "the Act",
        id="cfr-447.522(e)-investigational-drug-rebates",
    ),
    pytest.param(
        "us/regulation/42/423/2768",
        CFR_423_2768_G,
        ("g",),
        "1128A",
        "the Act",
        id="cfr-423.2768(g)-civil-money-penalties",
    ),
    pytest.param(
        "us/regulation/42/435/562",
        CFR_435_562_E,
        ("e",),
        "1904",
        "the Act",
        id="cfr-435.562(e)-corrective-action",
    ),
    pytest.param(
        "us-hi/regulation/har/17/1737/page-78",
        HAR_17_1737_DRUG_CATEGORIES,
        ("4",),
        "1927",
        "the Act",
        id="har-17-1737-excluded-drug-categories",
    ),
    pytest.param(
        "us-ky/policy/cms/1115-stc/teamky/page-10",
        CMS_1115_STC_AMENDMENT_PROCESS,
        ("6",),
        "1115",
        "the Act",
        id="cms-1115-stc-amendment-process",
    ),
    # Synthetic: a regulation quoting "this act" is still not an act itself.
    pytest.param(
        "us-ca/regulation/22/50555",
        "(a) Except as provided in section 5 of this act, the standard "
        "allowance is $100 for each month.",
        ("a",),
        "5",
        "this act",
        id="synthetic-regulation-quoting-this-act",
    ),
    # Synthetic: a manual that defines "the Act" as the Social Security Act.
    pytest.param(
        "us-ny/manual/otda/tanf/section-3",
        "(b) Unless otherwise provided in section 1902 of the Act, the income "
        "limit is 138 percent of the federal poverty level.",
        ("b",),
        "1902",
        "the Act",
        id="synthetic-manual-defined-the-act",
    ),
    # Synthetic: a statute that uses "the Act" as a defined external term.
    pytest.param(
        "us-ca/statute/wic/14005.30",
        "(a) Subject to section 1115 of the Act, the department shall provide "
        "coverage to persons whose income does not exceed 138 percent of the "
        "federal poverty level.",
        ("a",),
        "1115",
        "the Act",
        id="synthetic-statute-defined-the-act",
    ),
]

# Real self-references: a statute, session law or law compilation names a
# section of its own enactment.
SAME_ACT_CASES = [
    pytest.param(
        "us-ms/statute/27-7-5",
        "(1) For calendar year 2030 and all calendar years thereafter, except "
        "as otherwise provided in Section 2 of this act, on such taxable "
        "income, the rate shall be three percent (3%).",
        ("1",),
        "2",
        "this act",
        id="ms-27-7-5-session-law-rate",
    ),
    pytest.param(
        "us-nj/statute/43:21-61",
        "5. Except as provided in section 8 of this act, each individual who "
        "meets the requirements of section 4 of this act, but has not "
        "completed the remedial education and vocational training, shall be "
        "entitled to receive a weekly benefit equal to his previous weekly "
        "unemployment compensation benefit for each additional week.",
        ("5",),
        "8",
        "this act",
        id="nj-43:21-61-additional-benefits",
    ),
    pytest.param(
        "us-or/statute/constitution/block-23",
        "10. No person shall be eligible to receive money from said fund "
        "except the veterans as defined in Section 3 of this act.",
        ("10",),
        "3",
        "this act",
        id="or-veterans-fund-eligibility",
    ),
    pytest.param(
        "us-nj/statute/43:15a-98",
        "(d) Such a member shall thereafter be treated in all respects as a "
        "nonlaw enforcement officer member of the retirement system, except "
        "with regard to compulsory retirement as provided in section 3 of "
        "this act.",
        ("d",),
        "3",
        "this act",
        id="nj-43:15a-98-compulsory-retirement",
    ),
    pytest.param(
        "us-nj/statute/34:15-31.8",
        "7. Any injury, illness or death of a firefighter which may be caused "
        "by exposure to a known carcinogen shall be subject to the provisions "
        "of section 6 of this act.",
        ("7",),
        "6",
        "this act",
        id="nj-34:15-31.8-subject-to-provisions",
    ),
    pytest.param(
        "tz-znz/statute/act-2005-2/zanzibar-social-security-fund-2005/document-1",
        "21. Subject to section 25 of this Act, any surplus remaining in the "
        "Fund shall be held on deposit.",
        ("21",),
        "25",
        "this Act",
        id="tz-znz-subject-to-section",
    ),
    pytest.param(
        "us-wa/statute/session-laws/2023/chapter-475-essb-5187/page-1206",
        "(a) Replace the statewide forest practices permit database system. "
        "Funding is subject to the conditions, limitations, and review "
        "requirements of section 701 of this act; and (b) Provide a "
        "recommendation for ways that the forest products industry could help "
        "cover the cost of the new forest practice online system.",
        ("a",),
        "701",
        "this act",
        id="wa-2023-c475-budget-proviso",
    ),
    pytest.param(
        "us-pa/guidance/dli/uc-law/page-210",
        "(b) Payment of any bill rendered under subsection (a) shall be made "
        "not later than thirty days after such bill was mailed to the last "
        "known address of the nonprofit organization or was otherwise "
        "delivered to it, unless there has been an application for review and "
        "redetermination under section 301 of this act.",
        ("b",),
        "301",
        "this act",
        id="pa-uc-law-compilation-review",
    ),
]

REASON_TEMPLATES = (
    "{citation} depends on section {section} of {act}, which is not supplied.",
    "{citation} cannot be computed because section {section} of {act} is not "
    "available.",
)


def _branch_citation(corpus_citation_path: str, path: tuple[str, ...]) -> str:
    return corpus_citation_path + "".join(f"({part})" for part in path)


def _reasons(
    corpus_citation_path: str, path: tuple[str, ...], section: str, act: str
) -> list[str]:
    citation = _branch_citation(corpus_citation_path, path)
    return [
        template.format(citation=citation, section=section, act=act)
        for template in REASON_TEMPLATES
    ]


def _deferral_is_covered(
    corpus_citation_path: str,
    source_text: str,
    path: tuple[str, ...],
    reason: str,
) -> bool:
    branches = recognize_source_structure(
        source_text,
        corpus_citation_path=corpus_citation_path,
    )
    assert path in {branch.path for branch in branches}
    anchor = completeness_module._deferral_anchor_bases(corpus_citation_path)[0]
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": corpus_citation_path},
            "deferred_outputs": [
                {
                    "output": f"{anchor}/{'/'.join(path)}#deferred_output",
                    "reason": reason,
                }
            ],
        },
        "rules": [],
    }
    covered, _issues = completeness_module._deferred_coverage(
        payload,
        corpus_citation_path=corpus_citation_path,
        source_text=source_text,
        branches=branches,
    )
    return path in covered


@pytest.mark.parametrize(
    ("corpus_citation_path", "source_text", "path", "section", "act"),
    NOT_SAME_ACT_CASES,
)
def test_same_act_reader_rejects_defined_act_outside_statutes(
    corpus_citation_path, source_text, path, section, act
):
    for reason in _reasons(corpus_citation_path, path, section, act):
        assert not completeness_module._reason_names_missing_same_act_dependency(
            reason,
            source_text,
            corpus_citation_path=corpus_citation_path,
            path=path,
        )
        assert not _deferral_is_covered(corpus_citation_path, source_text, path, reason)


@pytest.mark.parametrize(
    ("corpus_citation_path", "source_text", "path", "section", "act"),
    SAME_ACT_CASES,
)
def test_same_act_reader_keeps_statute_and_law_compilation_self_references(
    corpus_citation_path, source_text, path, section, act
):
    for reason in _reasons(corpus_citation_path, path, section, act):
        assert completeness_module._reason_names_missing_same_act_dependency(
            reason,
            source_text,
            corpus_citation_path=corpus_citation_path,
            path=path,
        )
        assert _deferral_is_covered(corpus_citation_path, source_text, path, reason)


def test_same_act_reader_accepts_the_act_in_reason_for_this_act_source():
    # The source must self-reference ("this act"); the reason may still name
    # that same section as "the act".
    source = (
        "(1) For calendar year 2030 and all calendar years thereafter, except "
        "as otherwise provided in Section 2 of this act, the rate shall be "
        "three percent (3%)."
    )
    reason = (
        "us-ms/statute/27-7-5(1) cannot be computed because Section 2 of the "
        "act is not available."
    )

    assert completeness_module._reason_names_missing_same_act_dependency(
        reason,
        source,
        corpus_citation_path="us-ms/statute/27-7-5",
        path=("1",),
    )
    assert _deferral_is_covered("us-ms/statute/27-7-5", source, ("1",), reason)


def test_same_act_reader_still_requires_this_act_in_statute_source():
    # A statute's "the Act" is a defined external statute, not its own
    # enactment, even when the reason calls it "this act".
    source = (
        "(a) Subject to section 1115 of the Act, the department shall provide "
        "coverage to persons whose income does not exceed 138 percent of the "
        "federal poverty level."
    )
    reason = (
        "us-ca/statute/wic/14005.30(a) depends on section 1115 of this act, "
        "which is not supplied."
    )

    assert not completeness_module._reason_names_missing_same_act_dependency(
        reason,
        source,
        corpus_citation_path="us-ca/statute/wic/14005.30",
        path=("a",),
    )


MS_27_7_5_RATE_SCHEDULE = (
    "(ii) For calendar year 2024 and all calendar years thereafter, the tax "
    "imposed under subparagraph (iii) of paragraph (a) of this subsection upon "
    "all taxable income of individuals in excess of Ten Thousand Dollars "
    "($10,000.00), shall be at the following rates:\n\n"
    "1. For calendar year 2024, on such taxable income, the rate shall be four "
    "and seven-tenths percent (4.7%);\n\n"
    "6. For calendar year 2029, on such taxable income, the rate shall be three "
    "and one-quarter percent (3.25%); and\n\n"
    "7. For calendar year 2030 and all calendar years thereafter, except as "
    "otherwise provided in Section 2 of this act, on such taxable income, the "
    "rate shall be three percent (3%).\n"
)

# Same-act references a branch does not own, or that no exception or
# condition governs. #1930 accepts every one; the reader must not.
NOT_OWNED_OR_NOT_GOVERNED_CASES = [
    # Parent scope: only item 7 names Section 2; the rate schedule is operative.
    pytest.param(
        "us-ms/statute/27-7-5",
        MS_27_7_5_RATE_SCHEDULE,
        ("1", "b", "ii"),
        "2",
        id="parent-ms-27-7-5(1)(b)(ii)-rate-schedule",
    ),
    pytest.param(
        "us-ms/statute/27-7-5",
        "(1) (a) Except as otherwise provided in this section, there is hereby "
        "assessed and levied a tax at the following rates:\n\n"
        + MS_27_7_5_RATE_SCHEDULE,
        ("1",),
        "2",
        id="parent-ms-27-7-5(1)-whole-subsection",
    ),
    # Scope bleed into RCW compiler notes.
    pytest.param(
        "us-wa/statute/82/82.87/82.87.040/block-1",
        "(4)(a) The tax imposed in this section applies to the sale or exchange "
        "of long-term capital assets owned by the taxpayer. [ 2025 c 421 s 101 ; "
        '2021 c 196 s 5 .] Notes: Contingency — 2021 c 196 ss 5 and 16: "(1) If '
        "a court of competent jurisdiction, in a final judgment not subject to "
        "appeal, adjudges section 5 of this act unconstitutional, or otherwise "
        'invalid, section 16 of this act takes effect."',
        ("4",),
        "5",
        id="bleed-wa-82.87.040(4)-rcw-notes",
    ),
    # Scope bleed into an unrecognized sibling budget item.
    pytest.param(
        "us-wa/statute/session-laws/2023/chapter-475-essb-5187/page-1067",
        "(19) $1,360,000 of the accident account—state appropriation and 1 "
        "$240,000 of the medical aid account—state appropriation are provided "
        "2 solely for the department of labor and industries to 3 establish "
        "behavioral health apprenticeship programs. 4 (22) $2,000,000 of the "
        "accident account—state appropriation is provided solely for "
        "functionality that allows for web-based 34 document uploading. This "
        "project is subject to the conditions, 35 limitations, and review "
        "provided in section 701 of this act.",
        ("19",),
        "701",
        id="bleed-wa-2023-c475-sibling-proviso",
    ),
    # Scope bleed into later session-law effective-date sections.
    pytest.param(
        "us-md/statute/session-laws/2024/chapter-717/page-99",
        "(3) the revenue attributable to this requirement shall be remitted to "
        "the Comptroller not later than September 30, 2024. SECTION 24. AND BE "
        "IT FURTHER ENACTED, That Section 2 of this Act shall take effect July "
        "1, 2024. SECTION 25. AND BE IT FURTHER ENACTED, That, except as "
        "provided in Section 24 of this Act, this Act shall take effect June 1, "
        "2024.",
        ("3",),
        "24",
        id="bleed-md-2024-c717-effective-date-section",
    ),
    # Loose trigger: "subject to" a tax, not to the referenced section.
    pytest.param(
        "us-md/statute/session-laws/2024/chapter-717/page-99",
        "(1) as provided in § 12–105 of the Tax – General Article, as enacted "
        "by Section 2 of this Act, all cigarettes and other tobacco products "
        "held in the State on or after July 1, 2024, shall be subject to the "
        "tax on cigarettes and other tobacco products as enacted under Section "
        "2 of this Act;",
        ("1",),
        "2",
        id="trigger-md-2024-c717-subject-to-the-tax",
    ),
    pytest.param(
        "us-ok/statute/68-711",
        "(a) Before any motor vehicle is operated on the public highways of "
        "this state, the operation of which is subject to the tax levied by "
        "subsection (b) of Section 3 of this act, such motor vehicles must be "
        "registered under this act with the Tax Commission.",
        ("a",),
        "3",
        id="trigger-ok-68-711-subject-to-the-tax",
    ),
    # Synthetic: "subject to appeal" does not govern the later reference.
    pytest.param(
        "us-ca/statute/rtc/17041",
        "(a) A final judgment not subject to appeal that adjudges section 5 of "
        "this act invalid shall be reported to the Franchise Tax Board.",
        ("a",),
        "5",
        id="trigger-synthetic-not-subject-to-appeal",
    ),
]


@pytest.mark.parametrize(
    ("corpus_citation_path", "source_text", "path", "section"),
    NOT_OWNED_OR_NOT_GOVERNED_CASES,
)
def test_same_act_reader_rejects_unowned_or_ungoverned_references(
    corpus_citation_path, source_text, path, section
):
    for reason in _reasons(corpus_citation_path, path, section, "this act"):
        assert not completeness_module._reason_names_missing_same_act_dependency(
            reason,
            source_text,
            corpus_citation_path=corpus_citation_path,
            path=path,
        )
        assert not completeness_module._reason_dependency_is_source_bound(
            reason,
            source_text,
            corpus_citation_path=corpus_citation_path,
            path=path,
        )


def test_same_act_reader_keeps_the_leaf_that_owns_the_reference():
    source = (
        "7. For calendar year 2030 and all calendar years thereafter, except as "
        "otherwise provided in Section 2 of this act, on such taxable income, "
        "the rate shall be three percent (3%)."
    )
    reason = (
        "us-ms/statute/27-7-5(1)(b)(ii)(7) cannot be computed because Section 2 "
        "of this act is not available."
    )

    assert completeness_module._reason_names_missing_same_act_dependency(
        reason,
        source,
        corpus_citation_path="us-ms/statute/27-7-5",
        path=("1", "b", "ii", "7"),
    )


# Review of #1933: headed sections and quoted definitions bound the branch,
# and a taxing verb or a clause break in the gap does not govern the reference.
REVIEW_1933_NEGATIVE_CASES = [
    pytest.param(
        "us-or/statute/constitution/block-23",
        "10. Furnishing forms; printing, office supplies and equipment; "
        "employes; payment of expenses Section 1. State empowered to lend "
        "credit to pay veterans' bonus; issuance of bonds. No person shall be "
        "eligible to receive money from said fund except the veterans as "
        "defined in Section 3 of this act.",
        ("10",),
        "3",
        id="f1-or-heading-entry-runs-into-section-1",
    ),
    pytest.param(
        "us-nj/statute/43:13-22.3",
        "(b) The repayment of any contributions shall be made within such "
        "period of time as shall be granted by the commission but at no time "
        'for a greater period than 3 years. "Member" means any present '
        "employee of the city who shall, subject to the provisions of section "
        "13 of this act, have the right to purchase credit.",
        ("b",),
        "13",
        id="f2-nj-repayment-runs-into-definitions",
    ),
    pytest.param(
        "us-ok/statute/68-711",
        "(a) Each motor vehicle subject to the tax levied pursuant to Section 3 "
        "of this act must be registered with the Tax Commission.",
        ("a",),
        "3",
        id="f3-subject-to-tax-levied-pursuant-to",
    ),
    pytest.param(
        "us-ok/statute/68-711",
        "(a) Each motor vehicle subject to the tax levied as provided in "
        "subsection (b) of Section 3 of this act must be registered.",
        ("a",),
        "3",
        id="f3-subject-to-tax-levied-as-provided-in",
    ),
    pytest.param(
        "us-ok/statute/68-711",
        "(a) Each motor vehicle, except a farm tractor, is operated on the "
        "highways and taxed under Section 3 of this act.",
        ("a",),
        "3",
        id="f3-except-noun-then-taxed-under",
    ),
    pytest.param(
        "us-ok/statute/68-711",
        "(a) Each motor vehicle, unless exempt, that is taxed under Section 3 "
        "of this act must be registered.",
        ("a",),
        "3",
        id="f3-unless-exempt-taxed-under",
    ),
    pytest.param(
        "us-ca/statute/rtc/17041",
        "(a) The credit is $100, except that the tax imposed under section 5 of "
        "this act is not reduced.",
        ("a",),
        "5",
        id="f3-except-that-tax-imposed-under",
    ),
]

REVIEW_1933_POSITIVE_CASES = [
    pytest.param(
        "us-ca/statute/rtc/17041",
        "(a) Except as provided in section 4 or section 5 of this act, the "
        "credit is $100.",
        ("a",),
        "5",
        id="except-as-provided-in-section-4-or-section-5",
    ),
    pytest.param(
        "us-ca/statute/rtc/17041",
        "(a) Subject to the provisions of paragraph (2) of subsection (b) of "
        "section 5 of this act, the credit is $100.",
        ("a",),
        "5",
        id="nested-paragraph-of-subsection-of-section",
    ),
    pytest.param(
        "us-la/statute/rs/47-297",
        "A. Except as provided in Section 5 of this Act, the credit is $100. "
        "B. The fee is $50.",
        ("A",),
        "5",
        id="uppercase-item-label-owns-reference",
    ),
]


@pytest.mark.parametrize(
    ("corpus_citation_path", "source_text", "path", "section"),
    REVIEW_1933_NEGATIVE_CASES,
)
def test_same_act_reader_rejects_review_1933_shapes(
    corpus_citation_path, source_text, path, section
):
    for reason in _reasons(corpus_citation_path, path, section, "this act"):
        assert not completeness_module._reason_dependency_is_source_bound(
            reason,
            source_text,
            corpus_citation_path=corpus_citation_path,
            path=path,
        )


@pytest.mark.parametrize(
    ("corpus_citation_path", "source_text", "path", "section"),
    REVIEW_1933_POSITIVE_CASES,
)
def test_same_act_reader_keeps_review_1933_governed_shapes(
    corpus_citation_path, source_text, path, section
):
    for reason in _reasons(corpus_citation_path, path, section, "this act"):
        assert completeness_module._reason_names_missing_same_act_dependency(
            reason,
            source_text,
            corpus_citation_path=corpus_citation_path,
            path=path,
        )


def test_same_act_own_text_stops_at_next_uppercase_item():
    own = completeness_module._same_act_branch_own_text(
        "A. The credit is $100. B. Except as provided in Section 5 of this "
        "Act, the fee is $50."
    )

    assert "Section 5" not in own
