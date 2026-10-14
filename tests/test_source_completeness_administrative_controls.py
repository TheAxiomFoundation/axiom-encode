"""US administrative condition sentences are not source-stated controls.

An `if` / `when` / `unless` sentence whose main clause only administers the
program (a notice, a verification step, a Federal Register publication duty, an
agency workflow step or a form instruction) decides neither eligibility nor an
amount. On US paths it no longer asks for a principal derived output
("Source-stated control") or for paired positive/blocking test cases. The same
holds for the guidance-portal disclaimer, the superseded pre-OBBB history and
the attachment glossary, which the paired-case gate already treated as
non-operative but which still created control demands.

A sentence stays a control when it states a numeric limit or a computation, or
when anything after its leading condition states an outcome: eligibility, an
amount, a count, an exclusion, a deduction or allowance, coverage, a denial,
a termination or a penalty. Non-US sources keep every condition sentence, and
US manuals keep their numeric-limit scope (#1833).

The administrative sentences below are from 7 CFR 273.2, 273.4, 273.10 and
273.12, the FNS OBBB alien-eligibility memo, the Kansas K-40ES voucher and
C.R.S. 39-22-2003; the kept sentences include the review-round cases from the
same sources (7 CFR 273.2(f), 273.12, 273.21 and CMS CIB 2025-12-08).
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

US_PATHS = (
    "us/regulation/7/273/10",
    "us/guidance/usda/fns/snap-obbb-alien-eligibility-implementation-memo",
    "us/statute/42/415/i",
    "us-ks/guidance/department-of-revenue/forms/2026/k-40es/document-1",
)
NON_US_PATHS = (
    "uk/guidance/dwp/adm/chapter-1",
    "uk-harrow/manual/council-tax-reduction-scheme-2026-2027",
)

ADMINISTRATIVE = (
    # 7 CFR 273.10(g)(1): notices of approval and denial.
    "If an application is approved, the State agency shall provide the "
    "household with written notice of the amount of the allotment and the "
    "beginning and ending dates of the certification period.",
    "If the application is denied, the State agency shall provide the household "
    "with written notice explaining the basis for the denial, the household's "
    "right to request a fair hearing, and the telephone number of the SNAP "
    "office.",
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
    # FNS guidance-portal disclaimer, history, workflow and glossary.
    "Guidance documents lack the force and effect of law, unless expressly "
    "authorized by statute or incorporated into a contract.",
    "USDA may not cite, use, or rely on any guidance that is not available "
    "through their guidance portal, except to establish historical facts.",
    "Prior to the OBBB, certain lawfully present aliens were eligible to "
    "receive SNAP benefits, provided they met all other SNAP eligibility "
    "requirements and completed a 5-year waiting period, unless exempted by "
    "PRWORA.",
    "State agencies must review household circumstances to take appropriate "
    "action when this occurs and follow program rules for acting on changes.",
    # Federal Register publication duty.
    "If the Secretary revises the table, the Secretary shall publish the "
    "revised table in the Federal Register.",
    # Kansas K-40ES form instructions.
    "When the due date falls on a Saturday, Sunday, or legal holiday, "
    "substitute the next regular workday.",
    "If married filing joint, include both names and social security numbers (SSNs).",
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
    # 7 CFR 273.2(f)(1)(iv): a verification step joined to an allowance rule.
    "If the household wishes to claim expenses for an unoccupied home, the "
    "State agency shall verify the household's actual utility expenses for the "
    "unoccupied home in every case and shall not use the standard utility "
    "allowance.",
    # CMS CIB 2025-12-08: a notice joined to a denial or termination.
    "The state must then provide written notice and fair hearing rights in "
    "accordance with 42 C.F.R. part 431 subpart E (including advance notice in "
    "the case of an eligibility termination or other adverse action) and, if "
    "there is no other basis for Medicaid eligibility, deny the application or "
    "terminate eligibility by the end of the month following the 30-day period.",
    # 7 CFR 273.12(c): a notice whose content recalculates the allotment.
    "However, if a household fails to provide information regarding the "
    "associated changes in shelter costs within 10 days of the report, the "
    "State agency should send a notice to the household that their allotment "
    "will be recalculated without the deduction.",
    # 7 CFR 273.21: a notice whose content terminates the household.
    "This notice shall also advise the household that, if a complete report is "
    "not filed timely, the household will be terminated.",
    # 26 U.S.C. 32(d)(1) and 25A(g)(1): filing conditions on a credit.
    "In the case of an individual who is married, this section shall apply only "
    "if a joint return is filed for the taxable year under section 6013.",
    "No credit shall be allowed under subsection (a) to an individual unless "
    "the individual includes on the return of tax the individual's taxpayer "
    "identification number.",
    # A verification step that states a numeric limit.
    "Verification of income is required if income exceeds $500.",
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


@pytest.mark.parametrize("citation", US_PATHS)
@pytest.mark.parametrize("source", ADMINISTRATIVE)
def test_administrative_sentence_is_not_a_control_on_us_paths(
    source: str, citation: str
):
    assert not _controls(source, citation)


@pytest.mark.parametrize("citation", NON_US_PATHS)
@pytest.mark.parametrize("source", ADMINISTRATIVE)
def test_administrative_sentence_stays_a_control_outside_us(source: str, citation: str):
    assert _controls(source, citation)


@pytest.mark.parametrize("citation", US_PATHS)
@pytest.mark.parametrize("source", OPERATIVE)
def test_operative_condition_stays_a_control_on_us_paths(source: str, citation: str):
    assert _controls(source, citation)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        *((source, True) for source in ADMINISTRATIVE),
        *((source, False) for source in OPERATIVE),
        # `, then` ends a leading condition before its first comma does.
        (
            "If, however, the unclear information does not meet the criteria, "
            "then the State agency shall not act on the information.",
            True,
        ),
        # The guard reads everything after the leading condition.
        (
            "If the household voluntarily reports a change in its medical "
            "expenses, the State agency shall verify the change if the change "
            "would increase the household's allotment.",
            False,
        ),
        # A sentence without an administrative action is never administrative.
        ("When the household moves, update the address.", False),
        ("", False),
    ),
)
def test_source_condition_is_administrative(source: str, expected: bool):
    assert completeness_module._source_condition_is_administrative(source) is expected


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


def test_us_administrative_sentence_needs_no_paired_cases():
    # A candidate that encodes the source's limit is not asked for
    # positive/blocking cases for a notice sentence on the same page, while an
    # omitted operative exception on that page still is.
    regulation = "us-or/regulation/461/135/9999"
    notice = ADMINISTRATIVE[0]
    exception = OPERATIVE[0]

    assert not _paired_case_issues(notice, regulation)
    assert _paired_case_issues(exception, regulation)
    assert _paired_case_issues(notice, "uk/regulation/2013/376/9999")


def test_us_administrative_sentence_still_competes_for_case_pairs(monkeypatch):
    # The administrative sentence is hidden from the report, not from the
    # allocation of case pairs: a pair that the validator accepts for both a
    # procedural and an operative sentence on one path must not move to the
    # operative sentence only because the procedural one stopped competing.
    regulation = "us-or/regulation/461/135/9999"
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
