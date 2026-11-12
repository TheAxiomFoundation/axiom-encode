"""Federal US administrative condition sentences are not source-stated controls.

An `if` / `when` / `unless` sentence whose main verb only administers the
program (a notice, a verification step, a Federal Register publication duty, an
agency workflow step or a form instruction) decides neither eligibility nor an
amount. On federal US paths it no longer asks for a principal derived output
("Source-stated control") or appears in the paired positive/blocking case
report. The same holds for the guidance-portal disclaimer, the `State agencies
must review ...` workflow and a definitions lead-in with nothing after it.

A sentence stays a control when it states a numeric limit, a computation or a
claimant-facing outcome, when anything after its leading condition states an
outcome (eligibility, an amount, a count, an inclusion or exclusion, an
allowance or deduction, benefits, coverage, a denial, a closure, a termination,
an effective date or a penalty), or when the administrative word is not the
main verb (`the provider gives 2 weeks written notice of closure`, `E-Verify`).
State, manual and non-US sources keep every condition sentence they kept before
(US manuals keep their #1833 numeric-limit scope).

The administrative sentences below are from 7 CFR 273.2, 273.4, 273.10 and
273.12, the FNS OBBB alien-eligibility memo, the Kansas K-40ES voucher and
C.R.S. 39-22-2003. The kept sentences include the review rounds of #1893: the
7 CFR 273.2(f), 273.12, 273.21, CMS CIB 2025-12-08 and CMS SHO 26-001 tails,
and the operative sentences an independent corpus probe found (COMAR
13A.14.06.11F, O.C.G.A. 48-7-40.24, 9 CCR 2503-5 3.581 and 2503-6, Ohio Adm.Code
5101:4-3-13, 5101:4-4-09 and 5101:4-8-17, 55 Pa. Code 3042, 42 CFR 435.956 and
the CMS MA enrollment guidance).
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CONTROL = "[complete-source-unit:formula-output] Source-stated control"
PAIRED = "require paired positive/blocking cases"
EMPTY_CANDIDATE = "format: rulespec/v1\nmodule:\n  summary: probe\nrules: []\n"

FEDERAL_PATHS = (
    "us/regulation/7/273/10",
    "us/guidance/usda/fns/snap-obbb-alien-eligibility-implementation-memo",
    "us/statute/42/415/i",
)
# Federal forms, policies and rulemaking and every state source are not
# measured yet, so they keep every condition sentence.
OTHER_PATHS = (
    "us/form/irs/ty2025/i8962/block-95",
    "us/policy/irs/pub-15/2026/block-38",
    "us/rulemaking/fr/2026-12345/block-7",
    "us-ks/guidance/department-of-revenue/forms/2026/k-40es/document-1",
    "us-oh/regulation/agency-5101-4/chapter-5101-4-8/rule-5101-4-8-17",
    "uk/guidance/dwp/adm/chapter-1",
    "uk-harrow/manual/council-tax-reduction-scheme-2026-2027",
)

ADMINISTRATIVE = (
    # 7 CFR 273.10(g)(1) and (f)(5): notices.
    "If an application is approved, the State agency shall provide the "
    "household with written notice of the amount of the allotment and the "
    "beginning and ending dates of the certification period.",
    "If there is an individual or organization available that provides free "
    "legal representation, the notice shall also advise the household of the "
    "availability of the services.",
    "If the State agency extends a household's certification period, it must "
    "advise the household of the new certification ending date with a notice "
    "containing the same information as the notice of eligibility.",
    # 7 CFR 273.4(c)(5): who provides the sponsor information.
    "If the State agency subsequently receives information or verification, it "
    "must act on the information as a reported change in household membership "
    "in accordance with the timeliness standards in § 273.12 or § 273.21, as "
    "appropriate.",
    "If the sponsored alien refuses to cooperate in providing information or "
    "verification, other adult members of the alien's household are "
    "responsible for providing the information or verification required in "
    "accordance with the provisions of § 273.2(d).",
    # 7 CFR 273.2(f): verification of questionable factors.
    "If a member's citizenship or status as a non-citizen national is "
    "questionable, the State agency must verify the member's citizenship or "
    "non-citizen national status.",
    # 7 CFR 273.12(a): reminder notice.
    "If a household fails to file a complete report by the specified filing "
    "date, the State agency shall provide the household with a reminder notice.",
    # 7 CFR 273.12(c)(3): `, then` ends the condition.
    "If, however, the unclear information does not meet the criteria, then the "
    "State agency shall not act on the information.",
    # FNS guidance-portal disclaimer and workflow.
    "Guidance documents lack the force and effect of law, unless expressly "
    "authorized by statute or incorporated into a contract.",
    "USDA may not cite, use, or rely on any guidance that is not available "
    "through their guidance portal, except to establish historical facts.",
    "State agencies must review household circumstances to take appropriate "
    "action when this occurs and follow program rules for acting on changes.",
    # Federal Register publication duty.
    "If the Secretary revises the table, the Secretary shall publish the "
    "revised table in the Federal Register.",
    # A form instruction (Kansas K-40ES wording).
    "If you need assistance completing your vouchers, contact the Kansas "
    "Department of Revenue at 785-368-8222.",
    # C.R.S. 39-22-2003(1.7): a definitions lead-in with nothing after it.
    "(1.7) As used in this section, unless the context otherwise requires:",
)

OPERATIVE = (
    # The brief's negative control: a real omitted exception.
    "The household is eligible for the standard utility allowance unless the "
    "household contains an elderly member.",
    "A household may not receive the excess shelter deduction unless the "
    "household contains an elderly member.",
    # 7 CFR 273.4(c): outcomes behind documentation and consent conditions.
    "When a household indicates inability or unwillingness to provide "
    "documentation of alien status for any household member, the State agency "
    "must classify that member as an ineligible alien.",
    "If the sponsored alien fails to provide consent, he or she shall be ineligible.",
    "If the same sponsor is responsible for the entire household, the entire "
    "household is ineligible until such time as the household provides the "
    "needed sponsor information or verification.",
    # 7 CFR 273.10(g)(1)(ii): a notice that explains a denial stays a control.
    "If the application is denied, the State agency shall provide the household "
    "with written notice explaining the basis for the denial, the household's "
    "right to request a fair hearing, and the telephone number of the SNAP "
    "office.",
    # The superseded pre-OBBB history states an outcome (receive SNAP benefits).
    "Prior to the OBBB, certain lawfully present aliens were eligible to "
    "receive SNAP benefits, provided they met all other SNAP eligibility "
    "requirements and completed a 5-year waiting period, unless exempted by "
    "PRWORA.",
    # Review round 1 tails: 7 CFR 273.2(f)(1)(iv), CMS CIB 2025-12-08 and
    # CMS SHO 26-001 (also with `a 90-day ROP` and without the second sentence).
    "If the household wishes to claim expenses for an unoccupied home, the "
    "State agency shall verify the household's actual utility expenses for the "
    "unoccupied home in every case and shall not use the standard utility "
    "allowance.",
    "The state must then provide written notice and fair hearing rights in "
    "accordance with 42 C.F.R. part 431 subpart E (including advance notice in "
    "the case of an eligibility termination or other adverse action) and, if "
    "there is no other basis for Medicaid eligibility, deny the application or "
    "terminate eligibility by the end of the month following the 30-day period.",
    "If the beneficiary responds with additional information about their "
    "current immigration status, including a new declaration of satisfactory "
    "immigration status, and the state is unable to verify such status, the "
    "state must provide the beneficiary with a 90-day reasonable opportunity "
    "period (ROP), as required by sections 1137(d), 1902(a)(46), 1902(ee), "
    "1903(x), and 2105(c)(9) of the Act.41 If the state verifies that the "
    "beneficiary continues to have a satisfactory immigration status for the "
    "coverage in which they are enrolled, the beneficiary will retain coverage, "
    "and the state should notify the beneficiary that they continue to be "
    "eligible for the coverage in which they are enrolled.",
    "If the beneficiary responds with additional information about their "
    "current immigration status, including a new declaration of satisfactory "
    "immigration status, and the state is unable to verify such status, the "
    "state must provide the beneficiary with a 90-day ROP, as required by "
    "sections 1137(d), 1902(a)(46), 1902(ee), 1903(x), and 2105(c)(9) of the Act.",
    # 7 CFR 273.12(c) and 273.21: notice content that changes the allotment or
    # terminates the household.
    "However, if a household fails to provide information regarding the "
    "associated changes in shelter costs within 10 days of the report, the "
    "State agency should send a notice to the household that their allotment "
    "will be recalculated without the deduction.",
    "This notice shall also advise the household that, if a complete report is "
    "not filed timely, the household will be terminated.",
    # 26 U.S.C. 32(d)(1) and 25A(g)(1): filing conditions on a credit.
    "In the case of an individual who is married, this section shall apply only "
    "if a joint return is filed for the taxable year under section 6013.",
    "No credit shall be allowed under subsection (a) to an individual unless "
    "the individual includes on the return of tax the individual's taxpayer "
    "identification number.",
    # Numeric limits and budgeting rules in verification or workflow wording.
    "Verification of income is required if income exceeds $500.",
    "State agencies must use a standard medical deduction of $155 when the "
    "household's verified medical expenses exceed $35.",
    "State agencies must use the prior month's income when the household "
    "reports a change after the 10th of the month.",
    # Review round 2 (corpus probe): an administrative word that is not the
    # main verb, or a condition comma followed by `but`.
    "(iii) Except in cases of provider or family illness or death, the provider "
    "gives 2 weeks written notice of closure to the parent and the contractor.",
    "(1) “Business enterprise” or “taxpayer” means any "
    "enterprise or organization which is registered and authorized to use the "
    "federal employment verification system known as “E-Verify” and "
    "is engaged in or carrying on any business activities within this state, "
    "except that such term shall not include retail businesses.",
    "If continued misuse occurs (identified on the quarterly usage report after "
    "a warning has occurred), the cash grant payment portion of his or her EBT "
    "card shall be disabled for one month, requiring the county to notify the "
    "client of additional options for receipt of payment.",
    "If those identified in a-b above are not available, is a person who "
    "exercises responsibility for a dependent child within the person's home "
    "and provides verification of such responsibility such as a court order.",
    "(ii) When the individual has been satisfying the work requirement by "
    "participating in a SNAP E&T component, the date the county agency "
    "notifies the individual that they are no longer meeting the ABA work "
    "requirement.",
    "When the assistance group timely reported a change, but the county agency "
    "did not timely act on the change, the first month affected by the "
    "county's failure to act is to be the first month the county agency should "
    "have made the change effective.",
    "The period of disqualification shall be sent effective with the first "
    "allotment issued after the adverse notice period has expired, unless the "
    "assistance group requested a fair hearing and continued benefits.",
    "if the child is experiencing homelessness or is a foster child, then 90 "
    "calendar days from the date of enrollment to obtain up to date "
    "immunizations or provide documentation of exemption from the immunization "
    "requirement.",
    "(ii)(A) Ends on the earlier of the date the agency verifies the "
    "individual's citizenship or satisfactory immigration status or determines "
    "that the individual did not verify his or her citizenship or satisfactory "
    "immigration status in accordance with paragraph (a)(2) of this section, or "
    "90 days after the date described in paragraph (b)(2)(i) of this section, "
    "except that the agency may extend the period.",
    "When the disenrollment request is received and processed as outlined "
    "above, the election period and date the plan received a valid request to "
    "disenroll will determine the effective date of the disenrollment (refer to "
    "§ 30.8 for disenrollment effective dates).",
    # Review round 2 synthetic cases.
    "If the household fails to provide the requested verification, the State "
    "agency shall send a notice and close the case.",
    "The term household does not include an individual whose residence the "
    "State agency cannot verify, unless the individual is homeless.",
    "If the parent is absent from the home, the child is a deprived child "
    "(refer to section 3.604.2).",
    "If the household reapplies before the end of the month, the State agency "
    "shall provide benefits back to the date of the notice.",
    "The State agency shall not certify the household until it verifies the "
    "identity of the applicant, except for households entitled to expedited "
    "service.",
    "If the individual, or payer institution, verifies by writing the "
    "information on a separate document, the information is not tax "
    "information.",
    # Review round 2 (federal corpus re-probe): an operative tail after the
    # administrative verb, or a responsibility for services, not information.
    "If funds revert to a State, the State is responsible for providing "
    "low-income home energy assistance program services to the service "
    "population of the tribe.",
    "(ii) Except as provided in paragraph (g) of this section, notify the payee "
    "and backup withhold on reportable payments made to the account or "
    "accounts under the rules of paragraphs (d), (e), and (f) of this section.",
    "If there is a determination by the Secretary that the social security "
    "number provided to an eligible institution by a student is incorrect, and "
    "a loan has been guaranteed for such student, the institution shall notify "
    "and instruct the lender and guaranty agency to cease further disbursements "
    "of the loan, but such guaranty shall not be voided.",
    "If the beneficiary does not respond to this request for information, the "
    "State must send the notice of noncompliance to the beneficiary and provide "
    "the associated 30-calendar period for the beneficiary to make a "
    "satisfactory showing.",
    "When a party makes a request for payment, the Part D plan sponsor must "
    "notify the enrollee of its determination and make payment (when "
    "applicable) no later than 14 calendar days after receipt of the request.",
    "If any tax paid is refunded in whole or in part, the taxpayer shall notify "
    "the Secretary, who shall redetermine the amount of the tax for the year or "
    "years affected.",
    # Review round 3 (uncapped federal probe and adversarial cases): an `until`
    # tail, a `, then` after the main verb, a second predicate after the
    # administrative verb, and an operative tail after a `;`.
    "(vi) The individual must sign the election form, and the Part D sponsor "
    "will inform the individual when their participation in the program is "
    "active, and, until the individual receives that notification, they are not "
    "a participant in the program.",
    "If a Board member vacates his or her seat prior to the completion of his or "
    "her term, the Administrator will notify the Wireline Competition Bureau of "
    "such vacancy, and a successor will be chosen pursuant to the nomination and "
    "appointment process described in paragraph (c) of this section.",
    "If the household fails to provide the requested verification, the State "
    "agency shall terminate the household's participation, then send a notice "
    "to the household.",
    "If the household does not respond, the State agency shall close the case, "
    "then notify the household in writing.",
    "If the household does not return the form, deny the application, then send "
    "the household a notice.",
    "If the enrollee does not pay the premium by the end of the grace period, "
    "the agency must provide the enrollee with a notice and disenroll the "
    "enrollee.",
    "If the individual fails to comply without good cause, the State agency "
    "shall notify the individual and the individual shall be removed from the "
    "household.",
    "If the sponsor's income cannot be documented, the State agency shall send "
    "a notice of denial and the application shall be rejected.",
    "If the taxpayer fails to file a return, the Secretary shall notify the "
    "taxpayer and assess the tax shown on the return.",
    "If the corporation ceases to meet the requirements, the Secretary shall "
    "notify the corporation and the election shall be revoked.",
    "If an overissuance occurred, the State agency shall notify the household "
    "and the household must repay the overissuance.",
    "The State agency shall verify the household's gross income and shall not "
    "issue an allotment until verification is received, unless the household is "
    "a migrant household.",
    "When the household reports a change in income, the State agency must act "
    "on the change and recompute the household's allotment.",
    "If the household does not submit an application for recertification, the "
    "State agency shall send a notice of expiration and the certification "
    "lapses.",
    "If the debtor does not respond, the agency will notify the Secretary and "
    "offset the debt against any Federal payment to the debtor.",
    "If the application is approved, the agency shall provide the individual "
    "with a written notice and shall enroll the individual in a managed care "
    "plan.",
    "If the household fails to appear for the interview, the State agency shall "
    "send a notice of missed interview and the household must reapply.",
    "If the household fails to repay, the State agency shall notify the "
    "household and collect the claim by allotment reduction.",
    "If the household refuses to cooperate, the State agency shall notify the "
    "household and the household's participation shall stop.",
    "If the State agency finds a violation, it shall notify the vendor, "
    "terminating its participation in the program.",
    "If the household fails to appear for the interview, the State agency shall "
    "send a notice of missed interview; the household's application is then "
    "void.",
    "If the household's income, resources, or household size change, the State "
    "agency shall notify the household; and benefits are adjusted.",
    "If the State agency determines that the household is ineli- gible, the "
    "State agency shall send a notice.",
    # Review round 4: notice content with its own outcome (`notify the
    # individual that he or she may not participate`), outcome participles,
    # a trailing exception, a `follow the procedures` cross-reference that
    # imports a benefit rule, and an abbreviation inside a `;` tail.
    "If the individual is found to have committed an intentional Program violation, the State agency shall notify the individual that he or she may not participate in the Program.",
    "If the alien's sponsor refuses to sign the affidavit of support, the State agency shall advise the household that the alien is barred from receiving SNAP.",
    "If the household does not request a fair hearing, the State agency shall notify the household that the claim is established.",
    "If the household does not respond to the request for contact, the State agency shall send a notice of adverse action, and the reduction takes effect after the notice period expires.",
    "If the recipient does not request a waiver, we will notify the recipient that the recipient owes the overpayment.",
    "If the household fails to complete the interview, the State agency shall notify the household that it loses its expedited service.",
    "If the household refuses to cooperate with the quality control review, the State agency shall inform the household that the household will lose its certification.",
    "When a household member stops attending school, the State agency shall notify the household that the member is subject to the work requirements of section 273.7.",
    "If a State does not obligate its funds by the end of the fiscal year, the Secretary shall notify the State that the unobligated amount will be reallotted to other States.",
    "If the Lead Agency fails to submit the plan, the Secretary shall notify the Lead Agency that it forfeits its share of the discretionary funds.",
    "If the alien's immigration status cannot be verified, the agency shall send the alien a notice rejecting the application.",
    "The State agency shall verify the household's residency before certifying the household, except for households that are homeless.",
    "If the household reports that it will move out of the State, the State agency shall send the household a notice that its certification period is shortened.",
    "If the student fails to maintain satisfactory academic progress, the institution shall notify the student that the student will receive no further aid.",
    "If the taxpayer fails to make an installment when due, the Secretary shall notify the taxpayer that the installment agreement is in default.",
    "If the parent refuses to cooperate with child support enforcement, the State agency shall notify the family that its grant will be cut by the parent's share.",
    "If you do not give us the information we need, we will notify you that you can no longer receive SSI.",
    "If the household member was an adult at the time of the overissuance, the State agency shall notify the household that the member is responsible for the overissuance.",
    "The State agency shall verify the alien status of each applicant; if the status is not verified, the State agency shall deny the application.",
    "If the household refuses to provide its social security numbers, the State agency shall notify the household that it will not be certified.",
    "If the household fails to report, the State agency shall send a notice that benefits will be discontinued.",
    "If the overpayment is not waived, we will notify you that you are required to repay it.",
    "If the recipient fails to attend the orientation, send the recipient a notice reducing the grant.",
    "If the student does not meet an exemption, inform the student that the student may not receive SNAP.",
    "If the household's medical expenses are questionable, the State agency shall verify the expenses, unless the household contains an elderly member.",
    "If the household does not incur heating costs, the State agency shall notify the household that it does not get the standard utility allowance.",
    "If the individual does not return the renewal form, the agency must send a notice that coverage ends on the last day of the month.",
    "If the student's work hours cannot be verified, the State agency shall request verification, and the student is not a household member.",
    "If the household fails to comply, the State agency shall notify the household that the household is penalized.",
    "If the individual is disqualified, the State agency shall provide written notice; the period is one year for the first violation.",
    "If a household's failure to timely apply for recertification was due to "
    "an error of the State agency and therefore there was a break in "
    "participation, the State agency shall follow the procedures in "
    "\u00a7 273.14(e).",
    "(14) If a benefit overissuance is discovered for a month or months in "
    "which a mandatory E&T participant has already fulfilled a work component "
    "requirement, the State agency must follow the procedure specified in "
    "paragraph (m)(6)(v) of this section.",
    "If the sponsor does not sign, the State agency shall notify the "
    "household; under 7 U.S.C. 2014(i), the alien is ineligible.",
    "(3) The agency must inform beneficiaries and providers of the "
    "beneficiaries aggregate limit and notify beneficiaries and providers when "
    "a beneficiary has incurred out-of-pocket expenses up to the aggregate "
    "family limit and individual family members are no longer subject to cost "
    "sharing for the remainder of the family's current monthly or quarterly "
    "cap period.",
)


def _issues(content: str, source: str, citation: str, test_cases) -> list[str]:
    return list(
        analyze_complete_source_unit(
            content,
            source,
            corpus_citation_path=citation,
            test_cases=test_cases,
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text,
                profile="legacy",
            ),
            extract_numeric_grounding_occurrences=functools.partial(
                extract_typed_numeric_occurrences_from_text, profile="legacy"
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        ).issues
    )


def _controls(source: str, citation: str) -> list[str]:
    return [
        issue
        for issue in _issues(EMPTY_CANDIDATE, source, citation, [])
        if issue.startswith(CONTROL)
    ]


@pytest.mark.parametrize("citation", FEDERAL_PATHS)
@pytest.mark.parametrize("source", ADMINISTRATIVE)
def test_administrative_sentence_is_not_a_control_on_federal_paths(
    source: str, citation: str
):
    assert not _controls(source, citation)


@pytest.mark.parametrize("citation", OTHER_PATHS)
@pytest.mark.parametrize("source", ADMINISTRATIVE)
def test_administrative_sentence_stays_a_control_outside_federal_paths(
    source: str, citation: str
):
    assert _controls(source, citation)


@pytest.mark.parametrize("citation", FEDERAL_PATHS)
@pytest.mark.parametrize("source", OPERATIVE)
def test_operative_condition_stays_a_control_on_federal_paths(
    source: str, citation: str
):
    assert _controls(source, citation)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        *((source, True) for source in ADMINISTRATIVE),
        *((source, False) for source in OPERATIVE),
        # The guard reads everything after the leading condition.
        (
            "If the household voluntarily reports a change in its medical "
            "expenses, the State agency shall verify the change if the change "
            "would increase the household's allotment.",
            False,
        ),
        # A sentence without an administrative main verb is never administrative.
        ("When the household moves, update the address.", False),
        ("", False),
    ),
)
def test_source_condition_is_administrative(source: str, expected: bool):
    assert completeness_module._source_condition_is_administrative(source) is expected


@pytest.mark.parametrize(
    ("citation", "expected"),
    (
        ("us/regulation/7/273/10", True),
        ("us/guidance/irs/rev-proc-2025-32/page-14", True),
        ("us/statute/42/415/i", True),
        ("us/form/irs/ty2025/i8962/block-95", False),
        ("us/policy/irs/pub-15/2026/block-38", False),
        ("us/rulemaking/fr/2026-12345/block-7", False),
        ("us/manual/fns/snap-quality-control", False),
        ("us-ks/guidance/department-of-revenue/forms/2026/k-40es/document-1", False),
        ("us-co/statute/39/39-22-2003", False),
        ("uk/guidance/dwp/adm/chapter-1", False),
        ("", False),
    ),
)
def test_is_us_federal_rule_citation_path(citation: str, expected: bool):
    assert completeness_module._is_us_federal_rule_citation_path(citation) is expected


MANUAL_PAGE = "us-or/manual/odhs/open/page-644"


def test_us_manual_scope_is_unchanged():
    # Manual procedure was already outside the controls (#1833); a stated
    # numeric limit stays a control even in a verification sentence.
    assert not _controls(
        "When child care hours are requested due to schooling, verification of "
        "registration and current class schedule are required.",
        MANUAL_PAGE,
    )
    assert _controls(
        "Verification of income is required if income exceeds $500.", MANUAL_PAGE
    )


_LIMIT_CANDIDATE = """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: {CITATION}
  summary: probe
inputs:
- name: institution_meal_share_percent
  entity: Person
  dtype: Decimal
  period: Month
rules:
- name: institution_meal_share_limit_percent
  kind: parameter
  dtype: Decimal
  source: Institutional Residents
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: parameter
        source:
          corpus_citation_path: {CITATION}
          excerpt: more than 50% of their meals
  versions:
  - effective_from: '0001-01-01'
    formula: '50'
- name: institution_resident_ineligible
  kind: derived
  entity: Person
  dtype: Judgment
  period: Month
  source: Institutional Residents
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: condition
        source:
          corpus_citation_path: {CITATION}
          excerpt: Residents of institutions that provide more than 50% of their
            meals are not eligible.
  versions:
  - effective_from: '0001-01-01'
    formula: 'institution_meal_share_percent > institution_meal_share_limit_percent'
"""
_LIMIT_TESTS = """
- name: at limit
  period: 2026-05
  input:
    us-or:policies/odhs/open/page-999#input.institution_meal_share_percent: 50
  output:
    us-or:policies/odhs/open/page-999#institution_resident_ineligible: not_holds
- name: above limit
  period: 2026-05
  input:
    us-or:policies/odhs/open/page-999#input.institution_meal_share_percent: 51
  output:
    us-or:policies/odhs/open/page-999#institution_resident_ineligible: holds
"""
_LIMIT_SENTENCE = (
    "Residents of institutions that provide more than 50% of their meals are "
    "not eligible."
)


def _paired_case_issues(sentence: str, citation: str) -> list[str]:
    source = f"Institutional Residents\n\n{_LIMIT_SENTENCE}\n\n{sentence}\n"
    return [
        issue
        for issue in _issues(
            _LIMIT_CANDIDATE.replace("{CITATION}", citation),
            source,
            citation,
            yaml.safe_load(_LIMIT_TESTS),
        )
        if PAIRED in issue
    ]


def test_federal_administrative_sentence_needs_no_paired_cases():
    # A candidate that encodes the source's limit is not asked for
    # positive/blocking cases for a notice sentence on the same page, while an
    # omitted operative exception on that page still is.
    regulation = "us/regulation/7/273/9999"
    notice = ADMINISTRATIVE[0]
    exception = OPERATIVE[0]

    assert not _paired_case_issues(notice, regulation)
    assert _paired_case_issues(exception, regulation)
    assert _paired_case_issues(notice, "us-or/regulation/461/135/9999")
    assert _paired_case_issues(notice, "uk/regulation/2013/376/9999")


def test_federal_administrative_sentence_still_competes_for_case_pairs(monkeypatch):
    # The administrative sentence is hidden from the report, not from the
    # allocation of case pairs: a pair that the validator accepts for both a
    # procedural and an operative sentence on one path must not move to the
    # operative sentence only because the procedural one stopped competing.
    regulation = "us/regulation/7/273/9999"
    notice = ADMINISTRATIVE[0]
    allocated: list[str] = []
    allocate = completeness_module._unwitnessed_exception_branches

    def recording_allocate(branches, **kwargs):
        allocated.extend(branch.text for branch in branches)
        return allocate(branches, **kwargs)

    monkeypatch.setattr(
        completeness_module, "_unwitnessed_exception_branches", recording_allocate
    )

    assert not _paired_case_issues(notice, regulation)
    assert any(notice in text for text in allocated)


@pytest.mark.parametrize(
    ("citation", "source"),
    (
        # Line logic on federal forms and publications (review round 3): the
        # scope leaves them controls, and `enter your` is not an instruction.
        (
            "us/form/irs/ty2025/i8962/block-95",
            "Line 3. If you are not required to complete line 2b, enter your "
            "modified AGI from line 2a on line 3.",
        ),
        (
            "us/form/irs/ty2025/i1040gi/block-83",
            "If Exception 1 applies, enter your total capital gain distributions "
            "(from box 2a of Form(s) 1099-DIV) on line 7a and check the box "
            "Schedule D not required on line 7b.",
        ),
        (
            "us/form/irs/ty2026/p15/block-38",
            "If the employee later begins to work for another employer for wages "
            "subject to withholding, the employee must notify you within 10 days. "
            "You then figure withholding based on the daily or miscellaneous "
            "period.",
        ),
        (
            "us/guidance/irs/pub-590-a/2025/worksheet-1",
            "If married filing jointly, enter your modified AGI (that of both "
            "spouses) on line 2.",
        ),
    ),
)
def test_federal_form_line_logic_stays_a_control(citation: str, source: str):
    assert _controls(source, citation)
