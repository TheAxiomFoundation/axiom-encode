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
        "us-ok/statute/68-711",
        "(a) Before any motor vehicle is operated on the public highways of "
        "this state, the operation of which is subject to the tax levied by "
        "subsection (b) of Section 3 of this act, such motor vehicles must be "
        "registered under this act with the Tax Commission.",
        ("a",),
        "3",
        "this act",
        id="ok-68-711-special-fuel-registration",
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
