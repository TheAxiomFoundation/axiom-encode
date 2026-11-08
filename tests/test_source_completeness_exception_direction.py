"""Exception direction in the complete-source check.

The check pairs each exception or applicability condition in a source clause
with a case pair whose witness moves the output toward the pole the source
states: ``exclude`` moves it toward the negative pole (ineligible,
prohibited) and ``enable`` toward the positive pole (eligible, permitted).
Most of these tests model 12 CFR 1006.6 (Regulation F), whose prohibitions on
communications in connection with debt collection open with cross-references
and carry conditions of their own.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness import validator_pipeline as validator_pipeline_module
from tests.scaling import assert_linear_growth, lowest_work

REGULATION_F_1006_6 = (
    Path(__file__).parent / "fixtures" / "source_completeness" / "cfr_12_1006_6.txt"
)
CITATION = "us/regulation/12/1006/99"
MODULE = "us:regulations/12-cfr/1006/99"
KY_CITATION_PATH = "us-ky/statute/krs/141.020/document-1"
EN_NUMERIC_OCCURRENCES = functools.partial(
    validator_pipeline_module.extract_typed_numeric_inventory_occurrences_from_text,
    profile="en-US",
)
EN_NUMERIC_GROUNDING_OCCURRENCES = functools.partial(
    validator_pipeline_module.extract_typed_numeric_occurrences_from_text,
    profile="en-US",
)


def _requirement(text: str) -> str:
    return completeness_module._source_exception_effect_requirement(text)


def _capitalized(text: str) -> str:
    return text[0].upper() + text[1:]


# 12 CFR 1006.6(b)(3), (b)(4) and (c)(1), shortened where noted.
B3_BODY = (
    "a debt collector must not communicate or attempt to communicate with a "
    "consumer in connection with the collection of any debt at the consumer's "
    "place of employment, if the debt collector knows or has reason to know that "
    "the consumer's employer prohibits the consumer from receiving such "
    "communication."
)
B4_SHORTENED = (
    "The prohibitions in paragraphs (b)(1) through (3) of this section do not "
    "apply when a debt collector communicates with a consumer's attorney."
)
C1_SHORTENED = (
    "if a consumer notifies a debt collector in writing that the consumer refuses "
    "to pay a debt, the debt collector must not communicate further with the "
    "consumer."
)

ACCEPTANCE_ROWS = (
    pytest.param(
        "Except as provided in paragraph (b) of this section, a household is "
        "eligible if its income falls below the limit.",
        "enable",
        id="eligibility-if-after-cross-reference",
    ),
    pytest.param(
        "A household is eligible if its income falls below the limit.",
        "enable",
        id="eligibility-if",
    ),
    pytest.param(
        "Except as provided in paragraph (b) of this section, a debt collector may "
        "communicate with a consumer if the consumer consents.",
        "enable",
        id="permission-if-after-cross-reference",
    ),
    pytest.param(
        "A debt collector may communicate with a consumer if the consumer consents.",
        "enable",
        id="permission-if",
    ),
    pytest.param(
        "Except as provided in paragraph (c) of this section, " + B3_BODY,
        "exclude",
        id="1006.6(b)(3)-after-cross-reference",
    ),
    pytest.param(_capitalized(B3_BODY), "exclude", id="1006.6(b)(3)"),
    pytest.param(B4_SHORTENED, "enable", id="1006.6(b)(4)-do-not-apply"),
    pytest.param(
        B4_SHORTENED.replace("do not apply", "does not apply"),
        "enable",
        id="1006.6(b)(4)-does-not-apply",
    ),
    pytest.param(
        "A debt collector may not communicate with a consumer if the consumer objects.",
        "exclude",
        id="may-not-if",
    ),
    pytest.param(
        "A debt collector shall not communicate with a consumer if the consumer "
        "objects.",
        "exclude",
        id="shall-not-if",
    ),
    pytest.param(
        "Subject to paragraph (b) of this section, a debt collector may not "
        "communicate with a consumer if the consumer objects.",
        "exclude",
        id="may-not-if-after-subject-to",
    ),
    pytest.param(_capitalized(C1_SHORTENED), "exclude", id="1006.6(c)(1)"),
    pytest.param(
        "Except as provided in paragraph (c)(2) of this section, " + C1_SHORTENED,
        "exclude",
        id="1006.6(c)(1)-after-cross-reference",
    ),
    pytest.param(
        "A debt collector must not communicate with a consumer unless the consumer "
        "consents.",
        "enable",
        id="must-not-unless",
    ),
    pytest.param(
        "A debt collector must not communicate with a consumer except when the "
        "consumer consents.",
        "enable",
        id="must-not-except-when",
    ),
    pytest.param(
        "Except when the consumer consents, a debt collector must not communicate "
        "with a consumer.",
        "enable",
        id="preposed-except-when-must-not",
    ),
    pytest.param(
        "A person is not eligible unless the person files a return.",
        "enable",
        id="not-eligible-unless",
    ),
    pytest.param(
        "Unless the person files a return, a person is not eligible.",
        "enable",
        id="preposed-unless-not-eligible",
    ),
    pytest.param(
        "A household is eligible unless its income exceeds the limit.",
        "exclude",
        id="eligible-unless",
    ),
)


@pytest.mark.parametrize(("sentence", "requirement"), ACCEPTANCE_ROWS)
def test_requirement_follows_the_paired_conditions_own_cue(
    sentence: str, requirement: str
):
    """The requirement follows the paired condition's cue and the effect it qualifies.

    Invariant: a cross-reference that opens the sentence and states no
    condition ("Except as provided in paragraph (c) of this section,") sets no
    direction.  A condition of a prohibition excludes, a condition of a
    permission or an entitlement enables, and an "unless" or "except when"
    that opens the condition reverses the effect it qualifies.  "The
    prohibitions in paragraphs (b)(1) through (3) of this section do not apply
    when ..." (12 CFR 1006.6(b)(4)) lifts a prohibition, which permits, so its
    condition enables.
    """

    assert _requirement(sentence) == requirement


# Lead-ins whose reference cites the law at large, an official or agency, or
# a tariff heading.  26 U.S.C. 901(m)(3)(B) opens "Except as otherwise
# provided by the Secretary, in the case of ...".
LAW_AND_OFFICIAL_LEAD_INS = (
    "Except as otherwise provided by law, ",
    "Unless otherwise required by law, ",
    "Except as provided by statute, ",
    "Except as provided herein, ",
    "Except as otherwise provided below, ",
    "Except as provided in the preceding sentence, ",
    "Except as otherwise provided by the Secretary, ",
    "Except as authorized in writing by the Comptroller of the Currency, ",
    "Unless otherwise specified by FNS, ",
    "Except as provided for in Sec. 273.2, ",
    "Except as provided in heading 9903.88.01, ",
)
CONDITION_FREE_LEAD_INS = (
    "Except as provided in paragraph (b) of this section, ",
    "Except as otherwise provided in paragraph (c) of this section, ",
    "Except as provided in paragraphs (b), (c), and (d) of this section, ",
    "Except as specified in § 1006.14, ",
    "Except as provided by 42 U.S.C. 1382, ",
    "Unless otherwise provided in paragraph (b)(4) of this section, ",
    "Subject to paragraph (b) of this section, ",
    "Subject to the limitations in section 5 of this Act, ",
    *LAW_AND_OFFICIAL_LEAD_INS,
)
# (subject, effect phrase, direction of an "if" or "when" condition)
CLAUSE_EFFECTS = (
    ("a household", "is eligible", "enable"),
    ("the applicant", "is not eligible", "exclude"),
    ("the applicant", "is ineligible", "exclude"),
    ("a household", "shall be determined eligible", "enable"),
    ("the applicant", "shall not be considered eligible", "exclude"),
    (
        "an employee",
        "shall not be treated as eligible for minimum essential coverage",
        "exclude",
    ),
    ("a debt collector", "may communicate with a consumer", "enable"),
    ("a debt collector", "may send an email to the consumer", "enable"),
    ("a debt collector", "must not communicate with a consumer", "exclude"),
    ("a debt collector", "may not communicate with a consumer", "exclude"),
    ("a debt collector", "shall not communicate with a consumer", "exclude"),
    (
        "a debt collector",
        "must not place a telephone call to the consumer",
        "exclude",
    ),
    (
        "a debt collector",
        "is prohibited from communicating with a consumer",
        "exclude",
    ),
    ("the requirement in paragraph (a) of this section", "does not apply", "exclude"),
    ("the requirements in paragraph (a) of this section", "do not apply", "exclude"),
    (
        "the prohibitions in paragraphs (b)(1) through (3) of this section",
        "do not apply",
        "enable",
    ),
    ("the prohibition in paragraph (c)(1) of this section", "does not apply", "enable"),
)
CONDITION_CUES = ("if", "when")
REVERSAL_CUES = ("unless", "except when", "except if")
CONDITIONS = (
    "the consumer consents",
    "its income falls below the limit",
    "the consumer objects",
    "the debt is disputed",
)
REVERSED = {"enable": "exclude", "exclude": "enable"}


def _clause(subject: str, effect: str, cue: str, condition: str, preposed: bool):
    if preposed:
        return f"{cue} {condition}, {subject} {effect}."
    return f"{subject} {effect} {cue} {condition}."


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    effect=st.sampled_from(CLAUSE_EFFECTS),
    cue=st.sampled_from(CONDITION_CUES + REVERSAL_CUES),
    condition=st.sampled_from(CONDITIONS),
    preposed=st.booleans(),
    lead_in=st.sampled_from(CONDITION_FREE_LEAD_INS),
)
def test_condition_free_lead_in_leaves_the_requirement_unchanged(
    effect: tuple[str, str, str],
    cue: str,
    condition: str,
    preposed: bool,
    lead_in: str,
):
    """Lead-in invariance over a small clause grammar.

    Invariant: for a clause built from a subject, an effect (eligible, not
    eligible, a determination of either, a permission, a "must not", "may
    not", "shall not" or "is prohibited from" prohibition, a
    non-application, or a lifted prohibition), a cue and a condition, in
    either order, prefixing a cross-reference that states no condition
    leaves the requirement unchanged, and the requirement is the effect's
    own direction, reversed by "unless", "except when" or "except if".  This
    models 1006.6(b)(3) ("Except as provided in paragraph (b)(4) of
    this section, a debt collector must not communicate ... if the debt
    collector knows ...") and (c)(1) ("Except as provided in paragraph (c)(2)
    of this section, if a consumer notifies a debt collector ...").
    """

    subject, phrase, direction = effect
    clause = _clause(subject, phrase, cue, condition, preposed)
    expected = direction if cue in CONDITION_CUES else REVERSED[direction]

    assert _requirement(_capitalized(clause)) == expected
    assert _requirement(lead_in + clause) == expected


@pytest.mark.parametrize("lead_in", LAW_AND_OFFICIAL_LEAD_INS)
@pytest.mark.parametrize(
    ("clause", "requirement"),
    (
        ("if the consumer consents, a household is eligible.", "enable"),
        ("a household is eligible if the consumer consents.", "enable"),
        ("if the debt is disputed, the applicant is not eligible.", "exclude"),
        (
            "unless the consumer consents, a debt collector must not communicate "
            "with a consumer.",
            "enable",
        ),
    ),
)
def test_a_lead_in_that_cites_the_law_or_an_official_sets_no_direction(
    lead_in: str, clause: str, requirement: str
):
    """Lead-in invariance beyond provision numbers.

    Invariant: "Except as otherwise provided by law, if the consumer
    consents, a household is eligible." enables, as the clause does without
    its lead-in: a cross-reference to the law at large ("by law", "herein",
    "below"), to an official or agency ("by the Secretary", "by FNS") or to a
    tariff heading states no condition, so it sets no direction.
    """

    assert _requirement(_capitalized(clause)) == requirement
    assert _requirement(lead_in + clause) == requirement


CONDITION_FREE_REFERENCES = (
    "except as provided in paragraph (b) of this section",
    "except as otherwise provided in section 5",
    "unless otherwise provided in paragraph (c)",
    "subject to paragraph (b) of this section",
)


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    effect=st.sampled_from(CLAUSE_EFFECTS),
    reference=st.sampled_from(CONDITION_FREE_REFERENCES),
    cue=st.sampled_from(CONDITION_CUES + REVERSAL_CUES),
    condition=st.sampled_from(CONDITIONS),
)
def test_cross_reference_inside_the_clause_sets_no_direction(
    effect: tuple[str, str, str], reference: str, cue: str, condition: str
):
    """A cross-reference between an effect and its condition sets no direction.

    Invariant: "<Subject> <effect>, <reference>, <cue> <condition>." demands
    what "<Subject> <effect> <cue> <condition>." demands when the reference
    states no condition of its own: the effect's direction, reversed by
    "unless", "except when" or "except if".  Real text puts most of these
    references first, as 1006.6(b)(3) does; this extends the lead-in
    invariance to a reference set off by commas inside the clause, where an
    "except" right after a negative proposition could otherwise read as its
    exception.
    """

    subject, phrase, direction = effect
    expected = direction if cue in CONDITION_CUES else REVERSED[direction]
    plain = _capitalized(f"{subject} {phrase} {cue} {condition}.")
    referenced = _capitalized(f"{subject} {phrase}, {reference}, {cue} {condition}.")

    assert _requirement(plain) == expected
    assert _requirement(referenced) == expected


@pytest.mark.parametrize(
    ("sentence", "references"),
    [
        (
            "Except as provided in paragraphs (b), (c), and (d) of this section, a "
            "household is eligible.",
            ["Except as provided in paragraphs (b), (c), and (d) of this section"],
        ),
        (
            "Except as specified in \u00a7 1006.14, a debt collector must not "
            "communicate with a consumer.",
            ["Except as specified in \u00a7 1006.14"],
        ),
        (
            "Subject to paragraph (b) of this section, a household is eligible.",
            ["Subject to paragraph (b) of this section"],
        ),
        (
            "Unless otherwise provided in paragraph (c), a household is eligible.",
            ["Unless otherwise provided in paragraph (c)"],
        ),
        (
            "Except as provided in paragraph (b) when the household is elderly, a "
            "household is eligible.",
            [],
        ),
        (
            "Subject to the approval of the Secretary under section 5, a household "
            "is eligible.",
            [],
        ),
    ],
)
def test_cross_reference_spans_cover_whole_references(
    sentence: str, references: list[str]
):
    """A condition-free cross-reference spans its whole reference.

    Invariant: the span runs from the cue to the comma that closes the
    reference, past the commas of a list ("paragraphs (b), (c), and (d)").  A
    reference that states a condition ("when the household is elderly") is
    not condition-free, and neither is a "Subject to" phrase that names no
    provision ("the approval of the Secretary under section 5").
    """

    spans = completeness_module._source_cross_reference_spans(sentence)

    assert [sentence[start:end] for start, end in spans] == references


def _shall_not_apply_atom(excerpt: str) -> dict[str, object]:
    return {
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "kind": "exception",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        }
    }


NONAPPLICATION_READINGS = {
    "requirement": _requirement,
    "exception language": lambda text: bool(
        completeness_module._EXCEPTION_LANGUAGE.search(text)
    ),
    "unconditional non-application": (
        completeness_module._source_unconditional_nonapplicability
    ),
    "paired witness required": (
        completeness_module._source_exception_requires_paired_witness
    ),
    "boundary polarity": completeness_module._source_boundary_boolean_polarity,
    "negative effects": lambda text: len(
        completeness_module._source_negative_effect_matches(text)
    ),
    "negative output predicate": lambda text: bool(
        completeness_module._source_negative_output_predicate_matches(text)
    ),
    "shall-not-apply exception atom": lambda text: (
        validator_pipeline_module._rule_has_shall_not_apply_exception_atom(
            _shall_not_apply_atom(text)
        )
    ),
}


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    subject=st.sampled_from(
        (
            "the requirement in paragraph (a) of this section",
            "the requirements in paragraphs (a) and (b) of this section",
            "the prohibition in paragraph (c)(1) of this section",
            "the prohibitions in paragraphs (b)(1) through (3) of this section",
            "paragraphs (1) through (3)",
            "subsections (a) and (b)",
        )
    ),
    tail=st.sampled_from(
        (
            "",
            " if the consumer consents",
            " when the debt is disputed",
            " unless the consumer objects",
            " to a consumer's attorney",
        )
    ),
    lead_in=st.sampled_from(("",) + CONDITION_FREE_LEAD_INS),
    marker=st.sampled_from(("", "(1) ")),
)
def test_do_not_apply_reads_as_does_not_apply(
    subject: str, tail: str, lead_in: str, marker: str
):
    """Number agreement for non-application.

    Invariant: replacing "does not apply" with "do not apply" changes no
    reading that the complete-source check or the validator pipeline takes of
    a clause: the requirement, whether the exception language matches,
    whether the clause states an unconditional non-application, whether it
    needs a paired witness, its boundary polarity, its negative effects and
    negative output predicates, and whether an exception atom quoting it
    states a non-application.  1006.6(b)(4) uses the plural ("The prohibitions
    in paragraphs (b)(1) through (3) of this section do not apply when ...");
    (c)(2) uses the singular ("The prohibition in paragraph (c)(1) of this
    section does not apply when ...").
    """

    singular = marker + _capitalized(f"{lead_in}{subject} does not apply{tail}.")
    plural = singular.replace("does not apply", "do not apply")

    for name, read in NONAPPLICATION_READINGS.items():
        assert read(plural) == read(singular), name


@pytest.mark.parametrize(
    "source",
    [
        "(1) The provisions of sections 5 and 6 do not apply.",
        "(1) The requirements of section 5 do not apply.",
        "(1) Subsections (a) and (b) do not apply.",
        "(1) Paragraphs (1) through (3) do not apply.",
    ],
)
def test_plural_unconditional_nonapplicability_is_recognized(source: str):
    """A plural unconditional non-application needs no paired witness.

    Invariant: "Paragraphs (1) through (3) do not apply." states an
    unconditional non-application, as "Paragraphs (1) through (3) shall not
    apply." does, so the check asks for a false output rather than a case
    pair.
    """

    assert completeness_module._source_unconditional_nonapplicability(source)
    assert not completeness_module._source_exception_requires_paired_witness(source)


def test_plural_nonapplication_excerpt_is_a_shall_not_apply_atom():
    """An exception atom that quotes a plural non-application states one.

    Invariant: the validator pipeline reads an exception atom whose excerpt
    says "do not apply" as a non-application, as it reads "does not apply"
    and "shall not apply".  The excerpt is 1006.6(b)(4)'s lead-in.
    """

    rule = _shall_not_apply_atom(
        "The prohibitions in paragraphs (b)(1) through (3) of this section do not "
        "apply when a debt collector communicates or attempts to communicate with a "
        "consumer"
    )

    assert validator_pipeline_module._rule_has_shall_not_apply_exception_atom(rule)
    assert not validator_pipeline_module._rule_has_shall_not_apply_exception_atom(
        _shall_not_apply_atom("The prohibitions in paragraph (b)(1) apply.")
    )


ELIGIBILITY_ROWS_KEPT = (
    ("A household is eligible if its income falls below the limit.", "enable"),
    ("A household is eligible when its income falls below the limit.", "enable"),
    ("If its income falls below the limit, a household is eligible.", "enable"),
    (
        "Except as provided in paragraph (b) of this section, a household is "
        "eligible if its income falls below the limit.",
        "enable",
    ),
    ("The applicant is not eligible if the applicant owes a debt.", "exclude"),
    ("If the applicant owes a debt, the applicant is not eligible.", "exclude"),
    (
        "Except as provided in paragraph (b) of this section, the applicant is not "
        "eligible if the applicant owes a debt.",
        "exclude",
    ),
    ("A person is not eligible unless the person files a return.", "enable"),
    ("A household is eligible unless its income exceeds the limit.", "exclude"),
    ("(1) The claimant is ineligible unless an exception applies.", "enable"),
    ("(1) The claim is excluded except when a waiver applies.", "enable"),
    ("(1) The claimant is not allowed if a disqualification applies.", "exclude"),
    ("(1) The claimant is not entitled if a disqualification applies.", "exclude"),
    (
        "If the household moves, it is eligible for the shelter deduction but is "
        "not eligible for the utility allowance.",
        "exclude",
    ),
    (
        "If the household moves, it is not eligible for the utility allowance but "
        "is eligible for the shelter deduction.",
        "enable",
    ),
)


@pytest.mark.parametrize(("sentence", "requirement"), ELIGIBILITY_ROWS_KEPT)
def test_eligibility_requirements_are_stable(sentence: str, requirement: str):
    """Eligibility stability.

    Invariant: an eligibility clause keeps the requirement the check gave it
    before prohibitions, permissions and lifted prohibitions had a direction
    of their own: an "if" or "when" condition of an entitlement enables, one
    of an ineligibility excludes, and "unless" or "except when" reverses
    either.  A condition that opens a clause of eligibility effects alone
    takes the direction of the latest one.
    """

    assert _requirement(sentence) == requirement


DETERMINATIONS = (
    "determined",
    "found",
    "considered",
    "deemed",
    "treated as",
    "regarded as",
    "certified as",
)
# 7 CFR 273.2(d)(2), on a quality control reviewer.
QC_REFUSAL = (
    "If a household is terminated for refusal to cooperate with a quality "
    "control reviewer, in accordance with \u00a7\u00a7 275.3(c)(5) and "
    "275.12(g)(1)(ii) of this chapter, the household may reapply, but shall "
    "not be determined eligible until it cooperates with the quality control "
    "reviewer."
)
# 26 U.S.C. 36B(c)(2)(C)(ii).
MINIMUM_VALUE = (
    "Except as provided in clause (iii), an employee shall not be treated as "
    "eligible for minimum essential coverage if such coverage consists of an "
    "eligible employer-sponsored plan (as defined in section 5000A(f)(2)) and "
    "the plan\u2019s share of the total allowed costs of benefits provided "
    "under the plan is less than 60 percent of such costs."
)


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    determination=st.sampled_from(DETERMINATIONS),
    negated=st.booleans(),
    cue=st.sampled_from(CONDITION_CUES + REVERSAL_CUES),
    condition=st.sampled_from(CONDITIONS),
    preposed=st.booleans(),
)
def test_a_determination_of_eligibility_reads_as_eligibility(
    determination: str,
    negated: bool,
    cue: str,
    condition: str,
    preposed: bool,
):
    """Determination agreement.

    Invariant: "shall be determined eligible" reads as "is eligible", and
    "shall not be considered eligible" as "is not eligible", for each
    determination, cue and order: a determination of eligibility states
    eligibility.  This models 7 CFR 273.2(d)(2) ("the household may reapply,
    but shall not be determined eligible until it cooperates") and 26 U.S.C.
    36B(c)(2)(C)(ii) ("an employee shall not be treated as eligible for
    minimum essential coverage if ...").
    """

    plain = "is not eligible" if negated else "is eligible"
    determined = f"shall {'not ' if negated else ''}be {determination} eligible"
    direction = "exclude" if negated else "enable"
    expected = direction if cue in CONDITION_CUES else REVERSED[direction]

    for effect in (plain, determined):
        clause = _clause("the household", effect, cue, condition, preposed)
        assert _requirement(_capitalized(clause)) == expected


def test_permission_beside_a_determined_ineligibility_excludes():
    """7 CFR 273.2(d)(2) reads as its plain twin.

    Invariant: "the household may reapply, but shall not be determined
    eligible until it cooperates" excludes, as "the household may reapply,
    but is not eligible until it cooperates" does: the termination keeps the
    household ineligible, and the permission to reapply grants no
    eligibility.
    """

    plain = QC_REFUSAL.replace("shall not be determined eligible", "is not eligible")

    assert _requirement(QC_REFUSAL) == "exclude"
    assert _requirement(plain) == "exclude"


def test_determined_ineligibility_excludes_with_or_without_its_lead_in():
    """26 U.S.C. 36B(c)(2)(C)(ii) excludes with or without its cross-reference.

    Invariant: a plan below 60 percent of value makes the employee not
    eligible for minimum essential coverage, so its condition excludes; the
    lead-in "Except as provided in clause (iii)," sets none of that direction.
    """

    without_lead_in = _capitalized(MINIMUM_VALUE.split(", ", 1)[1])

    assert _requirement(MINIMUM_VALUE) == "exclude"
    assert _requirement(without_lead_in) == "exclude"


# 42 U.S.C. 1382(e)(1)(A), the SSI inmate exclusion.
SSI_INMATE = (
    "Except as provided in subparagraphs (B), (C), (D), (E), and (G), no "
    "person shall be an eligible individual or eligible spouse for purposes of "
    "this subchapter with respect to any month if throughout such month he is "
    "an inmate of a public institution."
)
ARTICLE_ROWS = (
    (SSI_INMATE, "exclude"),
    (_capitalized(SSI_INMATE.split("(G), ", 1)[1]), "exclude"),
    ("A person is not an eligible individual if the person is an inmate.", "exclude"),
    ("A person is an eligible individual if the person files a claim.", "enable"),
    (
        "No person shall be an eligible individual unless the person files a claim.",
        "enable",
    ),
    (
        "A household is a categorically eligible household if it receives TANF.",
        "enable",
    ),
)


@pytest.mark.parametrize(("sentence", "requirement"), ARTICLE_ROWS)
def test_an_article_before_eligible_keeps_the_eligibility_reading(
    sentence: str, requirement: str
):
    """42 U.S.C. 1382(e)(1)(A) excludes with or without its cross-reference.

    Invariant: inmate status throughout a month denies SSI eligibility, so
    its condition excludes: "no person shall be an eligible individual ... if"
    reads as "no person shall be eligible ... if" does.  An article before
    "eligible", with or without "categorically", leaves the statement of
    eligibility intact.
    """

    assert _requirement(sentence) == requirement


ARTICLE_FORMS = (
    "an eligible household",
    "a categorically eligible household",
)


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    form=st.sampled_from(ARTICLE_FORMS),
    modal=st.booleans(),
    negated=st.booleans(),
    cue=st.sampled_from(CONDITION_CUES + REVERSAL_CUES),
    condition=st.sampled_from(CONDITIONS),
    preposed=st.booleans(),
)
def test_an_article_before_eligible_reads_as_eligibility(
    form: str,
    modal: bool,
    negated: bool,
    cue: str,
    condition: str,
    preposed: bool,
):
    """Article agreement.

    Invariant: "is an eligible household" reads as "is eligible", and "shall
    not be a categorically eligible household" as "shall not be eligible",
    for each cue and order: an article before "eligible" leaves the
    statement of eligibility intact.
    """

    if modal:
        be = "shall not be" if negated else "shall be"
    else:
        be = "is not" if negated else "is"
    direction = "exclude" if negated else "enable"
    expected = direction if cue in CONDITION_CUES else REVERSED[direction]

    for effect in (f"{be} eligible", f"{be} {form}"):
        clause = _clause("the household", effect, cue, condition, preposed)
        assert _requirement(_capitalized(clause)) == expected


ARTICLE_CONDITION_EFFECTS = (
    "the household receives the credit",
    "the State agency shall deny the application",
    "the credit applies",
)


@pytest.mark.parametrize("effect", ARTICLE_CONDITION_EFFECTS)
@pytest.mark.parametrize("negated", (False, True))
@pytest.mark.parametrize("preposed", (False, True))
def test_an_article_before_eligible_agrees_with_the_plain_form_in_a_condition(
    effect: str, negated: bool, preposed: bool
):
    """Article agreement inside a condition.

    Invariant: "if the applicant is an eligible individual" sets the
    requirement that "if the applicant is eligible" sets, so the article
    changes no reading wherever the statement of eligibility sits.
    """

    be = "is not" if negated else "is"

    def sentence(condition: str) -> str:
        if preposed:
            return _capitalized(f"if {condition}, {effect}.")
        return _capitalized(f"{effect} if {condition}.")

    plain = sentence(f"the applicant {be} eligible")
    article = sentence(f"the applicant {be} an eligible individual")

    assert _requirement(article) == _requirement(plain)


# 45 CFR 1302.33(c)(4).
BILINGUAL_SCREENING = (
    "(4) If a program serves a child who speaks a language other than English "
    "and can demonstrate that there is not a qualified bilingual staff person "
    "or interpreter, then screenings and assessments may be conducted in "
    "English."
)


@pytest.mark.parametrize(
    "sentence",
    (
        BILINGUAL_SCREENING,
        "If there is an eligible child in the household, the household "
        "receives the credit.",
        "If there is not an eligible child in the household, the household "
        "receives the credit.",
    ),
)
def test_an_existential_there_keeps_its_article_out_of_the_reading(sentence: str):
    """45 CFR 1302.33(c)(4) sets no direction.

    Invariant: after "there is" or "there is not", "a qualified bilingual
    staff person" names something whose existence the condition asserts or
    denies, so it says nothing of whether the clause's subject qualifies, and
    the clause reads as it does without an article before "qualified".
    """

    assert _requirement(sentence) == "change"


# 26 U.S.C. 152(c)(4)(C), with its run-in heading.
NO_PARENT_CLAIMING = (
    "(C) No parent claiming qualifying child If the parents of an individual "
    "may claim such individual as a qualifying child but no parent so claims "
    "the individual, such individual may be claimed as the qualifying child of "
    "another taxpayer but only if the adjusted gross income of such taxpayer "
    "is higher than the highest adjusted gross income of any parent of the "
    "individual."
)


@pytest.mark.parametrize(
    "sentence",
    (NO_PARENT_CLAIMING, "If " + NO_PARENT_CLAIMING.split(" child If ", 1)[1]),
)
def test_a_heading_that_opens_with_no_states_no_prohibition(sentence: str):
    """26 U.S.C. 152(c)(4)(C) enables with or without its run-in heading.

    Invariant: the heading "No parent claiming qualifying child" is not the
    subject of the "may claim" in the condition after it, so the clause
    states no prohibition: when its conditions hold, another taxpayer may
    claim the individual.
    """

    assert _requirement(sentence) == "enable"


# 20 U.S.C. 1070a(j)(1), with its run-in heading.
INELIGIBLE_INSTITUTION = (
    "(1) In general No institution of higher education shall be an eligible "
    "institution for purposes of this subpart if such institution of higher "
    "education is ineligible to participate in a loan program under part B or "
    "D as a result of a final default rate determination made by the "
    "Secretary under part B or D after the final publication of cohort "
    "default rates for fiscal year 1996 or a succeeding fiscal year."
)


# 42 U.S.C. 1397ee(d)(1), with its parent's heading.
MEDICAID_MAINTENANCE = (
    "(d) Maintenance of effort (1) In medicaid eligibility standards No "
    "payment may be made under subsection (a) with respect to child health "
    "assistance provided under a State child health plan if the State adopts "
    "income and resource standards and methodologies for purposes of "
    "determining a child\u2019s eligibility for medical assistance under the "
    "State plan under subchapter XIX that are more restrictive than those "
    "applied as of June 1, 1997 , except as required under section "
    "1396a(e)(14) of this title ."
)


@pytest.mark.parametrize(
    "sentence",
    (
        INELIGIBLE_INSTITUTION,
        INELIGIBLE_INSTITUTION.split("In general ", 1)[1],
        "(j) Institutional ineligibility based on default rates "
        + INELIGIBLE_INSTITUTION,
    ),
)
def test_a_heading_with_lowercase_words_keeps_the_no_subject(sentence: str):
    """20 U.S.C. 1070a(j)(1) excludes with or without its run-in headings.

    Invariant: a run-in heading may carry lowercase words after its
    capitalized first word ("In general"), and a parent's heading and label
    may precede it; the "No" after them still opens the clause.  An
    institution that is ineligible for the loan programs is no eligible
    institution, so the condition excludes.
    """

    assert _requirement(sentence) == "exclude"


@pytest.mark.parametrize(
    ("text", "no_subject"),
    (
        ("No debt collector may communicate with the consumer.", True),
        ("No State agency or employee may disclose the information.", True),
        ("Neither the debt collector nor the creditor shall communicate.", True),
        (
            "No debt collector who knows the consumer is represented may "
            "communicate with the consumer.",
            True,
        ),
        ("(e) Students No individual may receive benefits.", True),
        ("(1) In general No institution may participate.", True),
        (
            "(j) Institutional ineligibility based on default rates (1) In "
            "general No institution may participate.",
            True,
        ),
        (MEDICAID_MAINTENANCE, True),
        ("No report is required before the applicant may apply.", False),
        (NO_PARENT_CLAIMING, False),
        ("The debt collector may communicate with the consumer.", False),
    ),
)
def test_a_no_subject_must_be_the_modal_subject(text: str, no_subject: bool):
    """No-subject guard for modals.

    Invariant: "No" makes the first modal after it a prohibition only when
    the words between them form the modal's subject: a noun phrase, which may
    join nouns with "and", "or" or "nor" and carry a relative clause.  Run-in
    headings, a parent's among them, may precede the "No".
    """

    modal = min(
        text.find(f" {word} ") for word in ("may", "shall") if f" {word} " in text
    )

    assert (
        completeness_module._source_modal_has_no_subject(text, modal + 1) is no_subject
    )


# 7 CFR 273.2(n)(4)(i)(A).
AUTHORIZED_REPRESENTATIVE_APPROVAL = (
    "State agency employees who are involved in the certification or issuance "
    "processes and retailers who are authorized to accept SNAP benefits may "
    "not act as authorized representatives without the specific written "
    "approval of a designated State agency official and only if that official "
    "determines that no one else is available to serve as an authorized "
    "representative."
)
WITHOUT_ROWS = (
    (AUTHORIZED_REPRESENTATIVE_APPROVAL, "enable"),
    (
        "A debt collector must not communicate with a consumer without prior "
        "approval, and only when the consumer consents.",
        "enable",
    ),
    (
        "No debt collector may communicate with a consumer without prior "
        "approval and only if the consumer consents.",
        "enable",
    ),
    (
        "A debt collector is prohibited from communicating with a consumer "
        "without prior approval, but only if the consumer consents.",
        "enable",
    ),
    (
        "A debt collector may not communicate with a consumer only if the "
        "consumer objects.",
        "exclude",
    ),
    (
        "A debt collector may not communicate with a consumer without prior "
        "approval if the consumer objects.",
        "exclude",
    ),
    (
        "A creditor acting without counsel may not sue the consumer only if the "
        "consumer objects.",
        "exclude",
    ),
    (
        "The school shall not transfer the equipment, with or without "
        "consideration, and only if the school closes.",
        "exclude",
    ),
    (
        "The household may not deduct costs computed without regard to income "
        "and only if it pays them.",
        "exclude",
    ),
)


@pytest.mark.parametrize(("sentence", "requirement"), WITHOUT_ROWS)
def test_an_only_if_that_joins_a_lifting_without_permits(
    sentence: str, requirement: str
):
    """7 CFR 273.2(n)(4)(i)(A) enables.

    Invariant: "may not act ... without the specific written approval ... and
    only if that official determines ..." permits the conduct with the
    approval and on the determination, so the "only if" condition enables.
    An "only if" that no lifting "without" follows the prohibition into keeps
    its plain reading, and "with or without" and "without regard to" lift
    nothing.
    """

    assert _requirement(sentence) == requirement


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    (
        (
            "Unless the household cooperates, the household is ineligible, but "
            "the household may reapply.",
            "enable",
        ),
        (
            "Unless the household cooperates, the household is ineligible, and "
            "the household may not reapply.",
            "enable",
        ),
        (
            "Unless the consumer consents, a debt collector may not communicate "
            "with the consumer, but the creditor may.",
            "enable",
        ),
        (
            "Unless the household cooperates, the household, and each member of "
            "it, is ineligible.",
            "enable",
        ),
    ),
)
def test_a_leading_cue_qualifies_its_own_clause(sentence: str, requirement: str):
    """Leading-cue scope.

    Invariant: "Unless the household cooperates, the household is
    ineligible, but the household may reapply" enables, as the clause does
    without its "but" clause: a leading cue qualifies the effect of the
    clause its condition opens, not a coordinate clause after it.  A comma
    inside that clause ("the household, and each member of it, is
    ineligible") does not end it.
    """

    assert _requirement(sentence) == requirement


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    (
        (
            "No person who is eligible shall be denied benefits if the person applies.",
            "enable",
        ),
        (
            "No individual who is enrolled shall be eligible if the individual moves.",
            "exclude",
        ),
    ),
)
def test_no_negates_the_effect_after_its_relative_clause(
    sentence: str, requirement: str
):
    """Relative-clause scope.

    Invariant: in "No person who is eligible shall be denied benefits", "No"
    negates the denial, not the eligibility that the relative clause states,
    so applying enables the benefits; in "No individual who is enrolled shall
    be eligible", it negates the eligibility after the relative clause.
    """

    assert _requirement(sentence) == requirement


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    (
        (
            "No person is prohibited from communicating with the consumer if the "
            "consumer consents.",
            "enable",
        ),
        (
            "No person is prohibited from communicating with the consumer when "
            "the consumer consents.",
            "enable",
        ),
        (
            "A person is prohibited from communicating with the consumer if the "
            "consumer objects.",
            "exclude",
        ),
        (
            "No person shall be prohibited from communicating with the consumer "
            "unless the consumer objects.",
            "exclude",
        ),
    ),
)
def test_no_subject_lifts_a_prohibition_from_conduct(sentence: str, requirement: str):
    """Lifted "prohibited from".

    Invariant: "No person is prohibited from communicating ... if the
    consumer consents" permits the communication when the consumer consents,
    so the condition enables; without "No", "is prohibited from" prohibits,
    and its condition excludes.
    """

    assert _requirement(sentence) == requirement


@pytest.mark.parametrize("comparative", ("sooner", "later", "more"))
def test_a_comparative_after_no_opens_a_limit(comparative: str):
    """Comparative limits.

    Invariant: "No sooner than 30 days after the denial the household may
    reapply if it cooperates" permits the reapplication, as "No later than"
    does: a comparative after "No" opens a limit, so "No" names no subject
    of the "may" that follows.
    """

    sentence = (
        f"No {comparative} than 30 days after the denial the household may "
        "reapply if it cooperates."
    )

    assert _requirement(sentence) == "enable"
    assert _requirement("No household may reapply if it refuses to cooperate.") == (
        "exclude"
    )


@pytest.mark.parametrize(
    "citation",
    (
        "15 U.S.C. 1692c",
        "12 C.F.R. 1006.6",
        "Pub. L. 95-109",
        "Pub. L. No. 95-109",
        "Sec. 805",
        "15 U.S.C. 1692 et seq.",
        "91 Stat. 874",
        "section 1692c",
    ),
)
def test_an_abbreviation_ends_no_sentence(citation: str):
    """Abbreviations.

    Invariant: "A debt collector must not communicate with a consumer under 15
    U.S.C. 1692c unless the consumer consents" enables, as it does with
    "section 1692c": the period of an abbreviation in a citation ends no
    sentence, so the cue still qualifies the prohibition before it.
    """

    sentence = (
        "A debt collector must not communicate with a consumer under "
        f"{citation} unless the consumer consents."
    )

    assert _requirement(sentence) == "enable"


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    (
        ("If the household applies in May benefits are prorated.", "change"),
        ("If the household applies by May 1, 2026 benefits are prorated.", "change"),
        ("(2) May communicate with the consumer if the consumer consents.", "enable"),
        ("May communicate with the consumer if the consumer consents.", "enable"),
    ),
)
def test_the_month_of_may_permits_nothing(sentence: str, requirement: str):
    """The month of May.

    Invariant: a capitalized "May" inside a clause names the month, so "If
    the household applies in May benefits are prorated" states no
    permission; a "May" that opens a clause or a list item keeps its modal
    reading.
    """

    assert _requirement(sentence) == requirement


# 20 U.S.C. 1070a(d)(6).
PELL_EXCLUSION = (
    "Beginning on July 1, 2026, and notwithstanding this subsection or "
    "subsection (b), a student shall not be eligible for a Federal Pell Grant "
    "under subsection (b) during any period for which the student receives "
    "grant aid from non-Federal sources, including States, institutions of "
    "higher education, or private sources, in an amount that equals or exceeds "
    "the student\u2019s cost of attendance for such period."
)
# 42 CFR 435.236(a)(1) and 435.622(a).
INSTITUTIONALIZED_GROUP = (
    "(1) Because of their income, would not be eligible for SSI or State "
    "supplements if they were not institutionalized;"
)
INSTITUTIONAL_INCOME_STANDARD = (
    "(a) If an agency, under \u00a7 435.231, provides Medicaid to individuals "
    "in medical institutions, nursing facilities, and intermediate care "
    "facilities for Individuals with Intellectual Disabilities who would not be "
    "eligible for SSI or State supplements if they were not institutionalized, "
    "the agency must use income standards based on the greater need for "
    "financial assistance that the individuals would have if they were not in "
    "the institution."
)


@pytest.mark.parametrize("auxiliary", ("shall", "will", "must", "is"))
def test_ruled_ineligibility_reads_as_is_not_eligible(auxiliary: str):
    """A rule's "shall not be eligible" reads as "is not eligible".

    Invariant: 20 U.S.C. 1070a(d)(6) pairs no cue, since "notwithstanding this
    subsection or subsection (b)" is a cross-reference, so its stated
    ineligibility excludes, whichever auxiliary states the rule.
    """

    effect = "is not eligible" if auxiliary == "is" else f"{auxiliary} not be eligible"
    ruled = PELL_EXCLUSION.replace("shall not be eligible", effect)

    assert _requirement(ruled) == "exclude"


@pytest.mark.parametrize(
    "sentence", (INSTITUTIONALIZED_GROUP, INSTITUTIONAL_INCOME_STANDARD)
)
def test_counterfactual_ineligibility_sets_no_direction(sentence: str):
    """A counterfactual ineligibility names a group and states no rule.

    Invariant: "would not be eligible for SSI ... if they were not
    institutionalized" says whom 42 CFR 435.236 covers, so it excludes
    nothing; in 435.622(a) the same words describe the group whose coverage
    requires an income standard, and they set no direction for it.
    """

    assert _requirement(sentence) == "change"


@pytest.mark.parametrize(
    "sentence",
    [
        "Unless the person files a return, a person is not eligible.",
        "Except when the person files a return, a person is not eligible.",
        "Except as provided in paragraph (b) of this section, if its income falls "
        "below the limit, a household is eligible.",
        "Subject to paragraph (b) of this section, if its income falls below the "
        "limit, a household is eligible.",
    ],
)
def test_preposed_eligibility_conditions_read_like_their_postposed_forms(
    sentence: str,
):
    """A preposed condition reads as the same condition placed after its effect.

    Invariant: "Unless the person files a return, a person is not eligible."
    reads as "A person is not eligible unless the person files a return.", and
    a cross-reference before a preposed "if" sets none of its direction.
    Each condition makes the person or the household eligible, so it enables.
    """

    assert _requirement(sentence) == "enable"


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    [
        (
            "Except as provided in paragraph (b) of this section, a person is not "
            "eligible.",
            "enable",
        ),
        (
            "Except as provided in paragraph (b)(4) of this section, a debt collector "
            "must not communicate with a consumer at any unusual time.",
            "enable",
        ),
        (
            "Unless otherwise provided in paragraph (c) of this section, a debt "
            "collector must not communicate with a consumer.",
            "enable",
        ),
    ],
)
def test_condition_free_cross_reference_alone_is_the_paired_condition(
    sentence: str, requirement: str
):
    """A cross-reference with no other cue is the condition the check pairs.

    Invariant: when nothing else in the clause states a condition, "Except as
    provided in paragraph (b) of this section" is the exception itself, and
    the exception it names reverses the clause's effect: it permits what a
    prohibition bars and makes eligible whom an ineligibility excludes.  This
    models 1006.6(b)(1), "Except as provided in paragraph (b)(4) of this
    section, a debt collector must not communicate ... at any unusual time".
    """

    assert _requirement(sentence) == requirement


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    [
        (
            "Subject to paragraph (b) of this section, a debt collector must not "
            "communicate with a consumer.",
            "change",
        ),
        (
            "Subject to paragraph (b) of this section, a person is not eligible.",
            "exclude",
        ),
    ],
)
def test_subject_to_reference_alone_keeps_its_reading(sentence: str, requirement: str):
    """A "Subject to" reference alone sets no direction of its own.

    Invariant: "Subject to paragraph (b) of this section" may narrow or widen
    the effect it qualifies, so a prohibition it alone qualifies demands no
    direction, and an ineligibility keeps the ``exclude`` that its own words
    ("is not eligible") demand.
    """

    assert _requirement(sentence) == requirement


# Effects this reading does not recognize, and conditions that carry
# negative words of their own.
UNREAD_EFFECTS = (
    "The following households are categorically eligible for benefits",
    "Households in which each member receives assistance shall be categorically "
    "eligible for benefits",
    "A household receives the standard deduction",
)
NEGATIVE_WORD_CONDITIONS = (
    "the entire household is disqualified",
    "the household is ineligible as specified in paragraph (b) of this section",
    "the household is not eligible under paragraph (c) of this section",
    "the member is excluded from the household",
)
NEUTRAL_CONDITIONS = (
    "the entire household is institutionalized",
    "the household moves out of the State",
)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    effect=st.sampled_from(UNREAD_EFFECTS),
    negative=st.sampled_from(NEGATIVE_WORD_CONDITIONS),
    neutral=st.sampled_from(NEUTRAL_CONDITIONS),
)
def test_trailing_condition_words_never_direct_the_pair(
    effect: str, negative: str, neutral: str
):
    """A trailing condition's own words are never the effect it qualifies.

    Invariant: when this reading recognizes no effect before a trailing
    "unless", the requirement of "<effect> unless <condition>" does not depend
    on whether the condition carries a negative word ("disqualified",
    "ineligible", "not eligible", "excluded"), and the "unless" demands an
    exclusion.  In 7 CFR 273.2(j)(2)(i), "The following households are
    categorically eligible for SNAP benefits unless the entire household is
    institutionalized ... or disqualified", "disqualified" belongs to the
    condition.
    """

    with_negative = _requirement(f"{effect} unless {negative}.")

    assert with_negative == _requirement(f"{effect} unless {neutral}.")
    assert with_negative == "exclude"


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    [
        (
            "Unless the household reports the change, the household is eligible.",
            "exclude",
        ),
        (
            "The State agency must act on the change unless the household reports "
            "it, and the household is eligible.",
            None,
        ),
    ],
)
def test_trailing_cue_never_reads_a_later_clause(sentence: str, requirement):
    """A trailing cue qualifies the clause before it, never the one after.

    Invariant: in "The State agency must act on the change unless the
    household reports it, and the household is eligible", the "must act"
    before "unless" shows that a clause precedes the cue.  The cue qualifies
    that clause, whose effect this reading does not recognize, so the reversal
    sets no direction; it does not borrow the eligibility of the clause after
    the comma.  "Unless the household reports the change, the household is
    eligible" has no clause before its cue, so the condition qualifies the
    eligibility after the comma, and the reversal excludes.
    """

    start = sentence.lower().index("unless")
    cue_span = (start, start + len("unless"))

    assert (
        completeness_module._source_effect_reversed_by_cue(sentence, cue_span)
        == requirement
    )


def test_qualification_idiom_keeps_its_reading_after_an_effect():
    """An effect before the idiom's "except" does not make it a reversal.

    Invariant: "The claimant who is eligible under paragraph (a) shall meet
    all qualifications, except the age requirement, in order to be eligible
    for the credit." keeps every qualification but one, so its exception
    enables, as the same rule without the relative clause does (the
    qualification test in test_source_completeness.py).  The "except" names
    the criterion the claimant need not meet; it does not reverse the "is
    eligible" before it.
    """

    sentence = (
        "The claimant who is eligible under paragraph (a) shall meet all "
        "qualifications, except the age requirement, in order to be eligible for "
        "the credit."
    )

    assert _requirement(sentence) == "enable"
    assert _requirement(
        sentence.replace(" who is eligible under paragraph (a)", "")
    ) == ("enable")


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    [
        (
            "No credit shall be allowed under this section unless the taxpayer "
            "includes the vehicle identification number on the return.",
            "enable",
        ),
        (
            "(9) VIN REQUIREMENT.—No credit shall be allowed under this section "
            "unless the taxpayer includes the vehicle identification number on the "
            "return.",
            "enable",
        ),
        (
            "No individual shall be eligible unless the individual files a return.",
            "enable",
        ),
        (
            "No household shall be eligible to participate if any member is on strike.",
            "exclude",
        ),
        (
            "Neither the applicant nor the spouse is eligible if either owes a debt.",
            "exclude",
        ),
        (
            "(e) Students No individual who is a member of a household otherwise "
            "eligible to participate shall be eligible to participate if the "
            "individual is enrolled at least half-time in an institution of higher "
            "education, unless the individual is employed for at least 20 hours a "
            "week.",
            "enable",
        ),
        (
            "No later than 30 days after the application, the household is eligible "
            "if its income falls below the limit.",
            "enable",
        ),
        (
            "No later than 30 days after the application the household is eligible "
            "if its income falls below the limit.",
            "enable",
        ),
        (
            "No later than 10 days after the request the State agency may hold a "
            "hearing if the household asks.",
            "enable",
        ),
        (
            "No fewer than two members of the household shall be eligible if the "
            "household applies.",
            "enable",
        ),
    ],
)
def test_no_subject_negates_an_entitlement(sentence: str, requirement: str):
    """A "No" subject makes an entitlement negative.

    Invariant: "No credit shall be allowed" and "No individual shall be
    eligible" state ineligibility, as "is not eligible" does, so an "if"
    condition of theirs excludes and an "unless" condition enables.  This
    models 7 U.S.C. 2015(e), whose student exceptions follow its "unless",
    and the vehicle identification number rule that Pub. L. 117-169 added to
    26 U.S.C. 30D.  A run-in heading ("(e) Students") may precede the
    subject, and "No later than" or "No fewer than" opens a limit, not a
    subject, with or without a comma after it.
    """

    assert _requirement(sentence) == requirement


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    [
        (
            "If a State agency has determined that an authorized representative has "
            "made improper use of benefits, it may disqualify that person from being "
            "an authorized representative.",
            "change",
        ),
        (
            "The State agency may terminate the agreement if the vendor violates it.",
            "change",
        ),
        (
            "If the household requests a waiver, the State agency may waive the "
            "interview requirement.",
            "change",
        ),
        (
            "The State agency may, upon request, waive the interview requirement if "
            "the household is elderly.",
            "change",
        ),
        (
            "The State agency may not deny the application if it caused the delay.",
            "change",
        ),
        (
            "The State agency shall not fail to notify the household if its benefits "
            "are reduced.",
            "change",
        ),
        (
            "The certification period shall not expire if the household reapplies.",
            "change",
        ),
        (
            "The State agency may limit the allotment if the household fails to "
            "cooperate.",
            "change",
        ),
        (
            "If the mail is delayed, circumstances may prevent an individual from "
            "receiving the notice.",
            "change",
        ),
        ("States may not know when an individual receives a notice.", "change"),
        (
            "If the notice is mailed, the household may expect it within five days.",
            "change",
        ),
        (
            "If the debt collector sends an email, it is possible that the consumer "
            "may see it.",
            "change",
        ),
        (
            "If the household is disabled, the State agency shall provide such "
            "assistance as the household may request.",
            "change",
        ),
        (
            "No debt collector may not communicate with a consumer if the consumer "
            "objects.",
            "change",
        ),
        (
            "States may not know when an individual receives a notice as mail "
            "delivery times vary or circumstances may prevent an individual from "
            "receiving their delivered mail.",
            "change",
        ),
        (
            "A debt collector may immediately communicate with a consumer if the "
            "consumer consents.",
            "enable",
        ),
        (
            "If the household requests one, the State agency may provide a "
            "home-based interview.",
            "enable",
        ),
    ],
)
def test_modal_direction_follows_its_verb(sentence: str, requirement: str):
    """A modal directs the pair only when its verb is conduct with a pole.

    Invariant: a permission to take something away ("may disqualify", "may
    terminate", "may limit") or to lift a requirement ("may waive"), and a
    prohibition of an adverse action ("may not deny"), can be encoded as the
    action (a positive output) or as the status it changes (a negative one),
    so neither direction follows and the pair need only change the output.
    "shall not fail to" obliges and "shall not expire" states a duration.
    "may prevent" can state a possibility and "may not know" a state of mind,
    as in the preamble to CMS-2454-IFC: "States may not know when an
    individual receives a notice as mail delivery times vary or circumstances
    may prevent an individual from receiving their delivered mail."  "may
    expect" states an expectation, and a "may" after "it is possible that" or
    in an "as the household may request" clause grants nothing either.  "No
    debt collector may not" negates the prohibition, so it obliges.  A phrase
    between the modal and its verb ("may, upon request, waive") leaves the
    verb unread.  An adverb ("may immediately communicate") does not: the
    permission still enables.
    """

    assert _requirement(sentence) == requirement


REPORTS = (
    "shall notify the household that",
    "must advise applicants that",
    "shall make clear that",
    "shall inform the consumer, in writing, that",
    "must explain to the household that",
)


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    effect=st.sampled_from(CLAUSE_EFFECTS),
    report=st.sampled_from(REPORTS),
    cue=st.sampled_from(CONDITION_CUES),
    condition=st.sampled_from(CONDITIONS),
)
def test_reported_effects_direct_nothing(
    effect: tuple[str, str, str], report: str, cue: str, condition: str
):
    """Effect words reported in a that-clause state no effect of the clause.

    Invariant: in "If C, the State agency shall notify the household that
    <subject> <effect>.", the reported words ("must not communicate", "may
    communicate", "do not apply") say what someone is told, not what the
    condition does, so the clause demands only a change.  This models 7 CFR
    273.2(c)(2)(i), "If the State agency attempts to discourage households
    from applying for cash assistance, it shall make clear that the
    disadvantages and requirements of applying for cash assistance do not
    apply to SNAP benefits."
    """

    subject, phrase, _ = effect
    sentence = (
        f"{_capitalized(cue)} {condition}, the State agency {report} {subject} "
        f"{phrase}."
    )

    assert _requirement(sentence) == "change"


def test_report_inside_a_condition_ends_at_its_comma():
    """A report inside the condition leaves the effect after it in force.

    Invariant: in 12 CFR 1006.6(c)(1), "if a consumer notifies a debt
    collector in writing that the consumer refuses to pay a debt, the debt
    collector must not communicate further", the notice's content ends at
    the comma, so the prohibition after it still demands an exclusion.
    """

    assert _requirement(_capitalized(C1_SHORTENED)) == "exclude"


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    effect=st.sampled_from(CLAUSE_EFFECTS),
    report=st.sampled_from(REPORTS),
    cue=st.sampled_from(CONDITION_CUES + REVERSAL_CUES),
    condition=st.sampled_from(CONDITIONS),
)
def test_reported_rule_that_carries_the_condition_directs_the_pair(
    effect: tuple[str, str, str], report: str, cue: str, condition: str
):
    """A condition inside a reported rule qualifies that rule.

    Invariant: in "The State agency <report> <subject> <effect> <cue>
    <condition>.", the clause's only condition sits inside the reported rule,
    so the clause demands what that rule demands on its own: the effect's
    direction, reversed by "unless", "except when" or "except if".
    """

    subject, phrase, direction = effect
    expected = direction if cue in CONDITION_CUES else REVERSED[direction]
    rule = f"{subject} {phrase} {cue} {condition}"

    assert _requirement(f"{_capitalized(rule)}.") == expected
    assert _requirement(f"The State agency {report} {rule}.") == expected


@pytest.mark.parametrize(
    "sentence",
    [
        "The State agency shall provide SSI households with a notice of expiration "
        "in accordance with § 273.14(b), except that such notification shall "
        "inform households consisting entirely of SSI recipients that they are "
        "entitled to a waiver of a face-to-face interview if the household is "
        "unable to appoint an authorized representative.",
        "(1) The notice shall inform the household member that if, at any time, "
        "he/she becomes exempt in accordance with Section 63-408.612 during a "
        "disqualification period, the disqualification shall end and he/she may "
        "reapply for food stamps.",
        "Notwithstanding subparagraph (A), the Secretary, informed by the report "
        "required under paragraph (2), may approve a plan that exempts "
        "participants or categories of participants if the Secretary determines "
        "that discontinuation of benefits in the form of cash is likely to have "
        "significant adverse effects.",
    ],
)
def test_reported_rules_keep_their_conditions(sentence: str):
    """A reported rule, commas included, carries the condition it states.

    Invariant: 7 CFR 273.2(k)(2)(i) has the notice tell SSI households
    "that they are entitled to a waiver ... if the household is unable to
    appoint an authorized representative", and California's notice rule
    (MPP 63-408.222) tells the member "that if, at any time, he/she becomes
    exempt ..., the disqualification shall end and he/she may reapply".  Each
    condition qualifies the entitlement its report states, so it enables.  In
    7 U.S.C. 2028(e)(3)(B) the "that" after "informed by the report ... may
    approve a plan" is a relative pronoun, not a report, so the permission
    keeps its condition and enables too.
    """

    assert _requirement(sentence) == "enable"


@pytest.mark.parametrize(
    ("sentence", "requirement"),
    [
        (
            "(3) The State agency may notify a household that its benefits will be "
            "reduced or terminated, no later than the date the household receives, "
            "or would have received, its allotment, if the following conditions are "
            "met:",
            "enable",
        ),
        (
            "States notify applicable individuals of their determination that a "
            "short-term hardship exception will or will not apply, and, if "
            "determining that it will, the date on which the exception will end.",
            "change",
        ),
    ],
)
def test_condition_after_a_reports_comma_qualifies_the_reporting_clause(
    sentence: str, requirement: str
):
    """A condition after a report's first comma belongs to the reporting clause.

    Invariant: a report that does not open with its condition ends at its
    first comma, so a later condition qualifies the clause that reports.  In
    7 CFR 273.13(a)(3) the conditions permit the State agency's notice, so
    they enable.  In the preamble to CMS-2454-IFC, "if determining that it
    will" qualifies the duty to give a date, and the reported "will not
    apply" sets no direction.
    """

    assert _requirement(sentence) == requirement


@pytest.mark.parametrize(
    ("condition", "effect"),
    [
        (
            "{det} rate of duty for which the article may be eligible is provided in "
            'the "Special" subcolumn',
            'the rate of duty provided in the "General" subcolumn shall apply',
        ),
        (
            "{det} member of the household is eligible for the program",
            "the household receives the standard deduction",
        ),
    ],
)
def test_no_inside_a_condition_negates_only_the_condition(condition: str, effect: str):
    """A "no" that opens a condition is not the subject of the clause.

    Invariant: "If no <noun> ..., <effect>." demands what "If a <noun> ...,
    <effect>." demands.  The "no" negates the condition, so it cannot turn the
    condition's own "may be eligible" into an ineligibility.  This models
    general note 3(a)(iii) of the Harmonized Tariff Schedule: "If no rate of
    duty for which the article may be eligible is provided in the 'Special'
    subcolumn ..., the rate of duty provided in the 'General' subcolumn shall
    apply."
    """

    negated = f"If {condition.format(det='no')}, {effect}."
    plain = f"If {condition.format(det='a')}, {effect}."

    assert _requirement(negated) == _requirement(plain)


PERMISSIONS = (
    "may communicate with the consumer",
    "may send an email to the consumer",
    "may follow up with the household",
)
PROHIBITIONS = (
    "must not communicate with the consumer",
    "may not communicate with the consumer",
    "shall not communicate with the consumer",
    "may not take adverse action",
)
JOINERS = (" but ", ", but ", " and ", ", and ", "; but ")
# 7 CFR 273.12(c)(3)(ii), second sentence.
FOLLOW_UP_RULE = (
    "A State may follow up with a household to provide information on a "
    "voluntary basis if that information would result in an increase in "
    "benefits but may not take adverse action if the household does not respond."
)


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    permission=st.sampled_from(PERMISSIONS),
    prohibition=st.sampled_from(PROHIBITIONS),
    conditions=st.permutations(CONDITIONS),
    cues=st.tuples(st.sampled_from(CONDITION_CUES), st.sampled_from(CONDITION_CUES)),
    joiner=st.sampled_from(JOINERS),
    permission_first=st.booleans(),
)
def test_permission_and_prohibition_in_one_clause_set_no_direction(
    permission: str,
    prohibition: str,
    conditions: list[str],
    cues: tuple[str, str],
    joiner: str,
    permission_first: bool,
):
    """A clause that permits on one condition and prohibits on another sets none.

    Invariant: in "A debt collector <permission> <cue> <condition> but
    <prohibition> <cue> <other condition>.", in either order and joined by
    "but" or "and", each condition has its own direction, so the one
    requirement the check pairs with the clause demands only a change.  This
    models 7 CFR 273.12(c)(3)(ii): "A State may follow up with a household to
    provide information on a voluntary basis if that information would result
    in an increase in benefits but may not take adverse action if the
    household does not respond."
    """

    permitted = f"{permission} {cues[0]} {conditions[0]}"
    prohibited = f"{prohibition} {cues[1]} {conditions[1]}"
    first, second = (
        (permitted, prohibited) if permission_first else (prohibited, permitted)
    )

    assert _requirement(f"A debt collector {first}{joiner}{second}.") == "change"


def test_follow_up_rule_sets_no_direction():
    """7 CFR 273.12(c)(3)(ii) permits on one condition and prohibits on another.

    Invariant: the follow-up the State may make "if that information would
    result in an increase in benefits" and the adverse action it may not take
    "if the household does not respond" pull opposite ways, so the clause
    demands only a change.
    """

    assert _requirement(FOLLOW_UP_RULE) == "change"


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    permission=st.sampled_from(PERMISSIONS),
    prohibition=st.sampled_from(PROHIBITIONS),
    condition=st.sampled_from(CONDITIONS),
    cue=st.sampled_from(CONDITION_CUES),
    joiner=st.sampled_from(JOINERS),
    permission_first=st.booleans(),
    preposed=st.booleans(),
)
def test_one_condition_of_a_permission_and_a_prohibition_sets_no_direction(
    permission: str,
    prohibition: str,
    condition: str,
    cue: str,
    joiner: str,
    permission_first: bool,
    preposed: bool,
):
    """A condition that both permits and prohibits sets no direction.

    Invariant: in "If <condition>, a debt collector <permission> but
    <prohibition>.", with the effects in either order, and with the condition
    after both of them, the one condition pulls both ways, so the clause
    demands only a change.  The order of the effects never decides it.
    """

    first, second = (
        (permission, prohibition) if permission_first else (prohibition, permission)
    )
    effects = f"{first}{joiner}{second}"
    sentence = (
        f"{_capitalized(cue)} {condition}, a debt collector {effects}."
        if preposed
        else f"A debt collector {effects} {cue} {condition}."
    )

    assert _requirement(sentence) == "change"


def _regulation_f_text() -> str:
    return " ".join(REGULATION_F_1006_6.read_text(encoding="utf-8").split())


def _regulation_f_clause(start: str, end: str) -> str:
    text = _regulation_f_text()
    begin = text.index(start)
    return text[begin : text.index(end, begin + len(start))].strip()


REGULATION_F_CLAUSES = (
    pytest.param(
        "Except as provided in paragraph (b)(4) of this section, a debt collector "
        "must not communicate or attempt to communicate with a consumer in "
        "connection with the collection of any debt:",
        " (i) At any unusual time",
        "enable",
        id="(b)(1)",
    ),
    pytest.param(
        "Except as provided in paragraph (b)(4) of this section, a debt collector "
        "must not communicate or attempt to communicate with a consumer in "
        "connection with the collection of any debt if the debt collector knows "
        "the consumer is represented",
        " (i) Fails to respond",
        "enable",
        id="(b)(2)",
    ),
    pytest.param(
        "Except as provided in paragraph (b)(4) of this section, a debt collector "
        "must not communicate or attempt to communicate with a consumer in "
        "connection with the collection of any debt at the consumer's place of "
        "employment",
        " (4) Exceptions.",
        "exclude",
        id="(b)(3)",
    ),
    pytest.param(
        "The prohibitions in paragraphs (b)(1) through (3) of this section do not "
        "apply",
        " (i) The prior consent",
        "enable",
        id="(b)(4)",
    ),
    pytest.param(
        "Except as provided in paragraph (c)(2) of this section, if a consumer "
        "notifies",
        " (2) Exceptions.",
        "exclude",
        id="(c)(1)",
    ),
    pytest.param(
        "The prohibition in paragraph (c)(1) of this section does not apply",
        " (i) To advise",
        "enable",
        id="(c)(2)",
    ),
    pytest.param(
        "Except as provided in paragraph (d)(2) of this section, a debt collector "
        "must not communicate",
        " (i) The consumer;",
        "enable",
        id="(d)(1)",
    ),
    pytest.param(
        "The prohibition in paragraph (d)(1) of this section does not apply",
        " (i) For the purpose",
        "enable",
        id="(d)(2)",
    ),
    pytest.param(
        "For purposes of paragraph (d)(3)(i) of this section, a debt collector may "
        "send an email",
        " (i) Procedures based",
        "enable",
        id="(d)(4)",
    ),
    pytest.param(
        "For purposes of paragraph (d)(3)(i) of this section, a debt collector may "
        "send a text message",
        " (i) The consumer used",
        "enable",
        id="(d)(5)",
    ),
)


@pytest.mark.parametrize(("start", "end", "requirement"), REGULATION_F_CLAUSES)
def test_regulation_f_1006_6_clauses_demand_their_stated_direction(
    start: str, end: str, requirement: str
):
    """12 CFR 1006.6, read from its corpus text.

    Invariant: each prohibition's own condition excludes ((b)(3), the
    employer's prohibition; (c)(1), the consumer's written notice); each
    exception to a prohibition enables, whether it opens the prohibition as
    a cross-reference ((b)(1), (d)(1)), follows it as "unless the attorney:"
    ((b)(2)), or lifts it with "do not apply" or "does not apply" ((b)(4),
    (c)(2), (d)(2)); and each permission's conditions enable ((d)(4), (d)(5)).
    """

    clause = _regulation_f_clause(start, end)

    assert clause.startswith(start)
    assert _requirement(clause) == requirement


def test_regulation_f_1006_6_conditions_without_their_lead_ins():
    """Dropping 1006.6's cross-reference lead-ins changes no direction.

    Invariant: (b)(3) and (c)(1) demand ``exclude`` with their "Except as
    provided in paragraph ... of this section," lead-ins and without them.
    """

    b3 = _regulation_f_clause(
        "Except as provided in paragraph (b)(4) of this section, a debt collector "
        "must not communicate or attempt to communicate with a consumer in "
        "connection with the collection of any debt at the consumer's place",
        " (4) Exceptions.",
    )
    c1 = _regulation_f_clause(
        "Except as provided in paragraph (c)(2) of this section, if a consumer",
        " (2) Exceptions.",
    )
    b3_lead_in = "Except as provided in paragraph (b)(4) of this section, a"
    c1_lead_in = "Except as provided in paragraph (c)(2) of this section, if"

    assert _requirement(b3) == "exclude"
    assert _requirement("A" + b3[len(b3_lead_in) :]) == "exclude"
    assert _requirement(c1) == "exclude"
    assert _requirement("If" + c1[len(c1_lead_in) :]) == "exclude"


def _missing_paired_witness(
    clause: str,
    *,
    output: str,
    selector: str,
    formula: str,
    active: bool,
) -> bool:
    """Run the complete-source check on one clause and a two-case pair."""

    source = (
        f"(a) Place of employment. {clause}.\n\n"
        "(b) Records. The debt collector must keep a record of each communication.\n"
    )
    document = {
        "format": "rulespec/v1",
        "module": {
            "summary": "Communication with a consumer.",
            "source_verification": {"corpus_citation_path": CITATION},
        },
        "inputs": [
            {"name": selector, "entity": "Person", "dtype": "Boolean", "period": "Day"}
        ],
        "rules": [
            {
                "name": output,
                "kind": "derived",
                "entity": "Person",
                "dtype": "Judgment",
                "period": "Day",
                "source": "12 CFR 1006.99(a)",
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "exception",
                                "source": {
                                    "corpus_citation_path": CITATION,
                                    "excerpt": clause,
                                },
                            }
                        ]
                    }
                },
                "versions": [{"effective_from": "0001-01-01", "formula": formula}],
            }
        ],
    }
    period = {
        "period_kind": "custom",
        "name": "day",
        "start": "2024-01-15",
        "end": "2024-01-15",
    }
    cases = [
        {
            "name": name,
            "period": period,
            "input": {f"{MODULE}#input.{selector}": value},
            "output": {f"{MODULE}#{output}": result},
        }
        for name, value, result in (
            ("output holds", active, "holds"),
            ("output does not hold", not active, "not_holds"),
        )
    ]
    content = yaml.safe_dump(document, sort_keys=False)
    bindings = completeness_module.collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=validator_pipeline_module.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    result = completeness_module.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        source_context={CITATION: source},
        test_cases=cases,
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCES,
        extract_numeric_grounding_occurrences=EN_NUMERIC_GROUNDING_OCCURRENCES,
        extract_named_scalars=validator_pipeline_module.extract_named_scalar_occurrences,
        numeric_value_is_grounded=validator_pipeline_module.numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return any("paired positive/blocking" in str(issue) for issue in result.issues)


PROBE_LEAD_INS = (
    "",
    "Except as provided in paragraph (c) of this section, ",
    "Except as otherwise provided in paragraph (c) of this section, ",
    "Subject to paragraph (c) of this section, ",
)
KNOWS = "collector_knows_or_has_reason_to_know_employer_prohibits_communication"
NEITHER_KNOWS = (
    "collector_neither_knows_nor_has_reason_to_know_employer_prohibits_communication"
)
DOES_NOT_KNOW = (
    "collector_does_not_know_or_have_reason_to_know_employer_prohibits_communication"
)
CONSENTS = "consumer_consents_to_communication"
WITHHOLDS_CONSENT = "consumer_withholds_consent_to_communication"
DOES_NOT_CONSENT = "consumer_does_not_consent_to_communication"


def _probe_clause(lead_in: str, body: str) -> str:
    return lead_in + body if lead_in else _capitalized(body)


def _prohibition_body(modal: str) -> str:
    return (
        f"a debt collector {modal} communicate or attempt to communicate with a "
        "consumer in connection with the collection of any debt at the consumer's "
        "place of employment, if the debt collector knows or has reason to know "
        "that the consumer's employer prohibits the consumer from receiving such "
        "communication"
    )


PERMISSION_BODY = (
    "a debt collector may communicate with a consumer in connection with the "
    "collection of any debt if the consumer consents to the communication"
)


@pytest.mark.parametrize("lead_in", PROBE_LEAD_INS)
@pytest.mark.parametrize("modal", ["must not", "may not", "shall not"])
def test_prohibition_pair_passes_only_when_faithful(lead_in: str, modal: str):
    """Polarity agreement for the 1006.6(b)(3) prohibition.

    Invariant: with or without a cross-reference lead-in, the check passes the
    faithful encoding of "a debt collector must not communicate ... if the
    debt collector knows or has reason to know that the consumer's employer
    prohibits ..." (the knowledge input switches
    ``employment_place_communication_prohibited`` on) and the same rule read
    through a negated input name it decodes (``not collector_does_not_know_...``).
    It rejects the encoding that negates an input named for the condition's
    absence in words it cannot decode (``not collector_neither_knows_nor_...``).
    """

    clause = _probe_clause(lead_in, _prohibition_body(modal))
    output = "employment_place_communication_prohibited"

    assert not _missing_paired_witness(
        clause, output=output, selector=KNOWS, formula=KNOWS, active=True
    )
    assert not _missing_paired_witness(
        clause,
        output=output,
        selector=DOES_NOT_KNOW,
        formula=f"not {DOES_NOT_KNOW}",
        active=False,
    )
    assert _missing_paired_witness(
        clause,
        output=output,
        selector=NEITHER_KNOWS,
        formula=f"not {NEITHER_KNOWS}",
        active=False,
    )


@pytest.mark.parametrize("lead_in", PROBE_LEAD_INS)
def test_permission_pair_passes_only_when_faithful(lead_in: str):
    """Polarity agreement for a permission, the prohibition probe's twin.

    Invariant: with or without a cross-reference lead-in, the check passes
    the faithful encoding of "a debt collector may communicate with a
    consumer ... if the consumer consents to the communication" (consent
    switches ``debt_collector_may_communicate`` on) and the same rule read
    through a decoded negated name (``not consumer_does_not_consent_...``).
    It rejects ``not consumer_withholds_consent_...``, which negates an input
    named for the condition's absence in words it cannot decode.
    """

    clause = _probe_clause(lead_in, PERMISSION_BODY)
    output = "debt_collector_may_communicate"

    assert not _missing_paired_witness(
        clause, output=output, selector=CONSENTS, formula=CONSENTS, active=True
    )
    assert not _missing_paired_witness(
        clause,
        output=output,
        selector=DOES_NOT_CONSENT,
        formula=f"not {DOES_NOT_CONSENT}",
        active=False,
    )
    assert _missing_paired_witness(
        clause,
        output=output,
        selector=WITHHOLDS_CONSENT,
        formula=f"not {WITHHOLDS_CONSENT}",
        active=False,
    )


@pytest.mark.parametrize(
    ("name", "negative"),
    [
        ("employment_place_communication_prohibited", True),
        ("communication_prohibited_at_place_of_employment", True),
        ("communication_prohibited_unless_consumer_consents", True),
        ("prohibited", True),
        ("place_of_employment_prohibition_applies", True),
        ("paragraph_b_prohibitions_apply", True),
        ("cease_communication_prohibition_is_applicable", True),
        ("prohibited_transaction_amount", False),
        ("prohibited_item_count", False),
        ("communication_not_prohibited", False),
        ("non_prohibited_communication", False),
        ("prohibition_does_not_apply", False),
        ("prohibition_exception_applies", False),
        ("paragraph_b_4_exception_applies", False),
        ("debt_collector_may_communicate", False),
        ("household_ineligible", True),
        ("household_eligible", False),
    ],
)
def test_output_name_polarity(name: str, negative: bool):
    """A name reads as negative when its predicate is a prohibition.

    Invariant: a prohibition is the negative pole of a permission, so
    ``employment_place_communication_prohibited`` (12 CFR 1006.6(b)(3)) and a
    name stating that a prohibition applies read as negative.  "prohibited"
    before the noun it modifies (``prohibited_transaction_amount``), a negated
    prohibition and an exception that applies read as non-negative.
    """

    assert (
        completeness_module._rule_has_negative_output_semantics(
            {"name": name}, fallback_name=name
        )
        is negative
    )


@pytest.mark.parametrize(
    ("description", "negative"),
    [
        (
            "A debt collector must not communicate with the consumer at the "
            "consumer's place of employment.",
            True,
        ),
        ("A debt collector may not communicate with the consumer.", True),
        ("No debt collector may communicate with the consumer.", True),
        ("A debt collector is prohibited from communicating with the consumer.", True),
        ("Communication at the place of employment is prohibited.", True),
        (
            "The prohibitions in paragraphs (b)(1) through (3) of this section apply.",
            True,
        ),
        ("A debt collector may communicate with the consumer.", False),
        (
            "The prohibition in paragraph (c)(1) of this section does not apply.",
            False,
        ),
        (
            "The prohibitions in paragraphs (b)(1) through (3) of this section do not "
            "apply.",
            False,
        ),
        ("The debt collector must comply with the restrictions that apply.", False),
        ("The household is eligible.", False),
        ("The household is not eligible if its income exceeds the limit.", True),
        ("No individual is eligible for the credit.", True),
        ("No credit shall be allowed for the vehicle.", True),
        (
            "The employee shall not be treated as eligible for minimum essential "
            "coverage.",
            True,
        ),
        ("The household shall be considered eligible.", False),
    ],
)
def test_output_description_polarity(description: str, negative: bool):
    """A description reads as negative when its output clause states a prohibition.

    Invariant: "A debt collector must not communicate with the consumer at the
    consumer's place of employment." states the prohibition of 12 CFR
    1006.6(b)(3), so the output is negative whatever its name; a lifted
    prohibition ("The prohibition in paragraph (c)(1) of this section does not
    apply.") permits, so it reads as positive.  A "No" subject negates an
    entitlement ("No credit shall be allowed"), so it reads as negative, and so
    does a determined ineligibility ("shall not be treated as eligible").
    """

    rule = {"name": "communication_output", "description": description}

    assert (
        completeness_module._rule_has_negative_output_semantics(
            rule, fallback_name="communication_output"
        )
        is negative
    )


# Output names for the 1006.6(b)(3) prohibition, read through the name alone.
# They cover the spellings the rulespec repositories use for prohibitions:
# participles ("barred", "forbidden"), negated permissions ("not_permitted")
# and negated modals ("must_not", "may_not", "shall_not", "cannot").
NEGATIVE_POLE_NAMES = (
    "employment_place_communication_prohibited",
    "employment_place_communication_barred",
    "employment_place_communication_forbidden",
    "employment_place_communication_not_permitted",
    "employment_place_communication_not_allowed",
    "debt_collector_must_not_communicate_at_place_of_employment",
    "debt_collector_may_not_communicate_at_place_of_employment",
    "debt_collector_shall_not_communicate_at_place_of_employment",
    "debt_collector_cannot_communicate_at_place_of_employment",
    "employment_place_communication_prohibited_due_to_employer_policy",
)
POSITIVE_POLE_NAMES = (
    "debt_collector_may_communicate_at_place_of_employment",
    "employment_place_communication_permitted",
    "employment_place_communication_not_barred",
)
UNREAD_POLE_NAMES = (
    "employment_place_communication",
    "employment_place_communication_status",
)


@pytest.mark.parametrize(
    ("name", "dtype", "pole"),
    [
        *((name, "Judgment", True) for name in NEGATIVE_POLE_NAMES),
        *((name, "Judgment", False) for name in POSITIVE_POLE_NAMES),
        *((name, "Judgment", None) for name in UNREAD_POLE_NAMES),
        # rulespec-us-ca regulations/mpp/63-401/6.yaml
        (
            "county_residency_requirement_barred_for_gain_work_supplemental_program_"
            "household",
            "Judgment",
            True,
        ),
        # rulespec-us us-co/regulations/10-ccr-2506-1/4.803.4.yaml
        (
            "administrative_hearing_request_prohibited_due_to_prosecution",
            "Judgment",
            True,
        ),
        ("qualified_non_citizen_barred_from_benefits", "Judgment", True),
        ("disqualification_without_written_verification_prohibited", "Judgment", True),
        (
            "remaining_members_snap_application_not_barred_by_ineligible_individual",
            "Judgment",
            False,
        ),
        ("household_not_determined_eligible", "Judgment", True),
        ("household_must_not_be_disqualified", "Judgment", False),
        ("fair_hearing_past_deadline_must_not_be_allowed", "Judgment", True),
        ("secretary_may_notify_payors_of_underreporting", "Judgment", False),
        ("household_will_be_eligible", "Judgment", False),
        ("household_will_receive_benefits", "Judgment", None),
        ("household_eligible_unless_disqualified", "Judgment", False),
        # rulespec-us-nc policies/dhhs/fns/fns-420-normal-application-
        # processing/page-2.yaml and rulespec-us-sc policies/dss/snap-policy-
        # manual/page-72.yaml
        (
            "deny_application_when_ineligible_by_interview_in_application_month",
            "Judgment",
            True,
        ),
        (
            "agency_must_deny_application_after_agency_delay_if_ineligible",
            "Judgment",
            True,
        ),
        ("agency_may_deny_application", "Judgment", None),
        # rulespec-us us-fl/policies/dcf/ess-program-policy-manual/1600-assets-
        # fs-tca/page-29.yaml
        ("tca_case_must_be_closed_after_unsold_excluded_property", "Judgment", True),
        ("case_must_be_closed", "Judgment", None),
        ("benefits_must_not_be_prorated", "Judgment", True),
        ("application_must_not_be_denied_on_thirtieth_day", "Judgment", None),
        ("household_may_be_determined_ineligible", "Judgment", True),
        ("household_may_be_determined_eligible", "Judgment", False),
        (
            "fns_unit_cannot_be_considered_categorically_eligible_after_lottery",
            "Judgment",
            True,
        ),
        ("household_may_be_considered_destitute", "Judgment", False),
        ("income_must_not_be_considered_available", "Judgment", None),
        # rulespec-us us-co/regulations/10-ccr-2506-1/4.411.2.yaml and
        # 4.203.1.yaml: the status names who the permission covers.
        (
            "remaining_members_may_be_certified_with_ineligible_student",
            "Judgment",
            False,
        ),
        ("ineligible_only_adult_may_be_head_and_apply", "Judgment", False),
        ("ineligible_household_count", "Judgment", None),
        ("household_not_disqualified", "Judgment", False),
        ("excluded_household_may_receive_utility_assistance", "Judgment", False),
        ("communication_barred_unless_consumer_consents", "Judgment", True),
        ("prohibited_employment_place_communication", "Judgment", True),
        ("prohibited_employment_place_communication", "Money", None),
        ("prohibited_transaction_amount", "Money", None),
        ("treasurer_may_prohibit_defaulting_bidder_from_bidding", "Judgment", None),
        ("income_may_not_exceed_limit", "Judgment", None),
        ("individual_or_group_shall_not_constitute_household", "Judgment", None),
        ("place_of_employment_prohibition_applies", "Judgment", True),
        ("prohibition_does_not_apply", "Judgment", False),
        ("restrictions_are_not_applicable", "Judgment", False),
        ("paragraph_b_4_exception_applies", "Judgment", None),
    ],
)
def test_output_pole_from_the_name(name: str, dtype: str, pole: bool | None):
    """A name states its pole through its predicate, or states none.

    Invariant: a prohibition participle as the predicate ("barred", "due to"
    after "prohibited"), a negated permission and a negated modal read as
    negative; a permission, an obligation and a negated prohibition read as
    positive.  A predicate word before a noun only modifies it, so
    ``qualified_non_citizen_barred_from_benefits`` reads by "barred", and a
    word in a condition clause (``..._unless_disqualified``) is not the
    output's.  A modal before a verb whose clause reading sets no direction
    (``may_prohibit``, ``may_not_exceed``) and a name with no predicate word
    state no pole, unless the name carries a disqualifying status, which
    then reads as negative wherever it sits
    (``deny_application_when_ineligible_...``).
    """

    rule = {"name": name, "dtype": dtype}

    assert completeness_module._rule_output_polarity(rule, fallback_name=name) is pole


DENIAL_CLAUSE = (
    "If the household is ineligible by the date of the interview held in the "
    "month of application, the county must deny the application"
)
CLOSURE_CLAUSE = (
    "The case must be closed if the excluded property is not sold during the "
    "nine month period"
)


@pytest.mark.parametrize(
    ("output", "clause", "selector"),
    (
        (
            "deny_application_when_ineligible_by_interview_in_application_month",
            DENIAL_CLAUSE,
            "household_ineligible_by_date_of_interview",
        ),
        (
            "agency_must_deny_application_after_agency_delay_if_ineligible",
            DENIAL_CLAUSE,
            "household_ineligible_by_date_of_interview",
        ),
        (
            "tca_case_must_be_closed_after_unsold_excluded_property",
            CLOSURE_CLAUSE,
            "excluded_property_not_sold_during_period",
        ),
    ),
)
def test_an_adverse_action_named_for_its_status_pairs_only_when_faithful(
    output: str, clause: str, selector: str
):
    """Polarity agreement for an output that names an adverse action.

    Invariant: each clause demands ``exclude`` of its condition, and each
    output's claim states no pole of its own ("deny the application", "the
    case must be closed") but its name carries a disqualifying status, so it
    reads as negative: the check passes the encoding that switches the
    action on when the condition holds and rejects the one that switches it
    off.
    """

    assert _requirement(clause + ".") == "exclude"
    assert not _missing_paired_witness(
        clause, output=output, selector=selector, formula=selector, active=True
    )
    assert _missing_paired_witness(
        clause,
        output=output,
        selector=selector,
        formula=f"not {selector}",
        active=False,
    )


@pytest.mark.parametrize(
    ("name", "pole"),
    [("utility_allowance_barred", True), ("utility_allowance_status", None)],
)
def test_output_pole_ignores_a_description_that_states_both_poles(
    name: str, pole: bool | None
):
    """A description that states both poles leaves the name to decide."""

    rule = {
        "name": name,
        "dtype": "Judgment",
        "description": (
            "The household is eligible for the shelter deduction but is not "
            "eligible for the utility allowance."
        ),
    }

    assert completeness_module._rule_output_polarity(rule, fallback_name=name) is pole


@pytest.mark.parametrize("lead_in", PROBE_LEAD_INS)
@pytest.mark.parametrize("output", NEGATIVE_POLE_NAMES)
def test_prohibition_pair_follows_a_negative_output_name(lead_in: str, output: str):
    """Polarity agreement for the 1006.6(b)(3) prohibition across its namings.

    Invariant: for an output whose name states the prohibition, the check
    passes the faithful pair (knowledge switches the prohibition on) and
    rejects the inverted one (knowledge switches it off), with or without a
    cross-reference lead-in.
    """

    clause = _probe_clause(lead_in, _prohibition_body("must not"))

    assert not _missing_paired_witness(
        clause, output=output, selector=KNOWS, formula=KNOWS, active=True
    )
    assert _missing_paired_witness(
        clause, output=output, selector=KNOWS, formula=f"not {KNOWS}", active=False
    )


@pytest.mark.parametrize("lead_in", PROBE_LEAD_INS)
@pytest.mark.parametrize("output", POSITIVE_POLE_NAMES)
def test_prohibition_pair_follows_a_positive_output_name(lead_in: str, output: str):
    """Polarity agreement when the output names the permission the rule bars.

    Invariant: knowledge switches ``debt_collector_may_communicate_...`` off,
    so the check passes that pair and rejects the one that switches it on.
    """

    clause = _probe_clause(lead_in, _prohibition_body("must not"))

    assert not _missing_paired_witness(
        clause, output=output, selector=KNOWS, formula=f"not {KNOWS}", active=False
    )
    assert _missing_paired_witness(
        clause, output=output, selector=KNOWS, formula=KNOWS, active=True
    )


@pytest.mark.parametrize("lead_in", PROBE_LEAD_INS)
@pytest.mark.parametrize("output", UNREAD_POLE_NAMES)
def test_prohibition_pair_sets_no_direction_for_an_unread_output(
    lead_in: str, output: str
):
    """An output that states no pole can name the prohibition or the permission.

    Invariant: when a prohibition sets the demand and neither the output's
    name nor its description states a pole, the check accepts a pair in
    either direction, as it did before it read directions for prohibitions.
    """

    clause = _probe_clause(lead_in, _prohibition_body("must not"))

    assert not _missing_paired_witness(
        clause, output=output, selector=KNOWS, formula=KNOWS, active=True
    )
    assert not _missing_paired_witness(
        clause, output=output, selector=KNOWS, formula=f"not {KNOWS}", active=False
    )


@pytest.mark.parametrize("lead_in", PROBE_LEAD_INS)
@pytest.mark.parametrize("output", UNREAD_POLE_NAMES)
def test_permission_pair_holds_an_unread_output_to_its_direction(
    lead_in: str, output: str
):
    """Only a prohibition frees an output that states no pole.

    Invariant: a permission's condition switches an output that states no
    pole on, as it always has, so the check rejects the pair that switches
    it off.
    """

    clause = _probe_clause(lead_in, PERMISSION_BODY)

    assert not _missing_paired_witness(
        clause, output=output, selector=CONSENTS, formula=CONSENTS, active=True
    )
    assert _missing_paired_witness(
        clause,
        output=output,
        selector=CONSENTS,
        formula=f"not {CONSENTS}",
        active=False,
    )


# 12 CFR 1006.6(c)(2), the lifted prohibition.
C2_LIFTED = (
    "The prohibition in paragraph (c)(1) of this section does not apply when a "
    "debt collector communicates further with a consumer with respect to such "
    "debt to advise the consumer that the debt collector's further efforts are "
    "being terminated"
)
ADVISES = (
    "debt_collector_communicates_further_to_advise_consumer_further_efforts_"
    "being_terminated"
)


@pytest.mark.parametrize(
    ("output", "pole"),
    [
        ("further_communication_prohibited", True),
        ("further_communication_barred", True),
        ("further_communication_forbidden", True),
        ("debt_collector_must_not_communicate_further", True),
        ("debt_collector_may_communicate_further", False),
        ("further_communication_permitted", False),
        ("further_communication_status", None),
    ],
)
def test_lifted_prohibition_pair_follows_the_output_pole(output: str, pole):
    """Polarity agreement for the lifted prohibition of 1006.6(c)(2).

    Invariant: advising the consumer lifts the prohibition, so the faithful
    pair switches a negative output off and a positive one on; the check
    rejects the other pair, and accepts both for an output that states no
    pole.
    """

    lifts = {"selector": ADVISES, "formula": f"not {ADVISES}", "active": False}
    keeps = {"selector": ADVISES, "formula": ADVISES, "active": True}
    faithful, inverted = (lifts, keeps) if pole is not False else (keeps, lifts)

    assert not _missing_paired_witness(C2_LIFTED, output=output, **faithful)
    assert _missing_paired_witness(C2_LIFTED, output=output, **inverted) is (
        pole is not None
    )


# California MPP 63-401.6, as rulespec-us-ca regulations/mpp/63-401/6.yaml
# encodes it.
CA_COUNTY_RESIDENCY = (
    "The CWD shall not require the otherwise eligible household to reside in the "
    "county which is providing the food stamp benefit if the household is "
    "participating in the GAIN Work Supplemental Program"
)


def test_barred_requirement_passes_its_faithful_pair():
    """The merged CA module's formula passes, and its inversion does not.

    Invariant: participation bars the residency requirement, so
    ``county_residency_requirement_barred_for_..._household = participating``
    is faithful.
    """

    output = (
        "county_residency_requirement_barred_for_gain_work_supplemental_program_"
        "household"
    )
    selector = "household_participating_in_gain_work_supplemental_program"

    assert not _missing_paired_witness(
        CA_COUNTY_RESIDENCY,
        output=output,
        selector=selector,
        formula=selector,
        active=True,
    )
    assert _missing_paired_witness(
        CA_COUNTY_RESIDENCY,
        output=output,
        selector=selector,
        formula=f"not {selector}",
        active=False,
    )


@pytest.mark.parametrize(
    ("clause", "states"),
    [
        (_capitalized(B3_BODY), True),
        (
            "A debt collector is prohibited from communicating with the consumer if "
            "the consumer objects.",
            True,
        ),
        (
            "Communication at the place of employment is prohibited if the employer "
            "objects.",
            True,
        ),
        (B4_SHORTENED, True),
        (
            "The prohibitions in paragraphs (b)(1) through (3) of this section apply "
            "when the consumer objects.",
            True,
        ),
        (
            "No debt collector may communicate with the consumer if the consumer "
            "objects.",
            True,
        ),
        (
            "The State agency must notify the household that, if it moves, it may "
            "not reapply.",
            True,
        ),
        (
            "The State agency must notify the household that it may not reapply if "
            "it moves.",
            True,
        ),
        (
            "A debt collector may communicate with the consumer if the consumer "
            "consents.",
            False,
        ),
        ("The household is ineligible if its income exceeds the limit.", False),
        ("The State agency must notify the household that it may not reapply.", False),
    ],
)
def test_clause_states_prohibition(clause: str, states: bool):
    """A clause that states, applies or lifts a prohibition sets a prohibition demand.

    Invariant: words reported in a that-clause state no prohibition of the
    clause unless they carry every condition of it, as the requirement reads
    them.
    """

    assert completeness_module._source_clause_states_prohibition(clause) is states


NAME_SUBJECTS = (
    "communication",
    "employment_place_communication",
    "household_application",
    "further_collection_action",
    "county_residency_requirement",
)
NAME_TAILS = (
    "",
    "_at_place_of_employment",
    "_for_household",
    "_due_to_prosecution",
    "_before_deadline",
    "_pending_hearing",
)
NAME_CONDITIONS = (
    "",
    "_unless_household_eligible",
    "_unless_consumer_disqualified",
    "_if_consumer_may_not_object",
    "_when_prohibited",
)
NEGATIVE_PREDICATE_WORDS = (
    "banned",
    "barred",
    "disallowed",
    "disqualified",
    "excluded",
    "forbidden",
    "ineligible",
    "precluded",
    "prohibited",
    "unqualified",
)
POSITIVE_PREDICATE_WORDS = (
    "allowable",
    "allowed",
    "authorized",
    "eligible",
    "entitled",
    "permissible",
    "permitted",
    "qualified",
    "required",
)


def _name_pole(name: str) -> bool | None:
    return completeness_module._rule_output_polarity(
        {"name": name, "dtype": "Judgment"}, fallback_name=name
    )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    subject=st.sampled_from(NAME_SUBJECTS),
    predicate=st.sampled_from(NEGATIVE_PREDICATE_WORDS + POSITIVE_PREDICATE_WORDS),
    tail=st.sampled_from(NAME_TAILS),
    condition=st.sampled_from(NAME_CONDITIONS),
)
def test_name_pole_reverses_with_not_and_ignores_its_conditions(
    subject: str, predicate: str, tail: str, condition: str
):
    """Invariants of the name's predicate word.

    The word sets the pole; "not" just before it reverses the pole; and a
    condition clause after the claim never changes it, whatever words the
    condition holds.
    """

    negative = predicate in NEGATIVE_PREDICATE_WORDS

    assert _name_pole(f"{subject}_{predicate}{tail}") is negative
    assert _name_pole(f"{subject}_not_{predicate}{tail}") is (not negative)
    assert _name_pole(f"{subject}_{predicate}{tail}{condition}") is negative


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    subject=st.sampled_from(
        ("debt_collector", "state_agency", "employer", "secretary")
    ),
    modal=st.sampled_from(("can", "may", "must", "shall")),
    verb=st.sampled_from(("communicate", "contact", "call", "visit", "notify")),
    tail=st.sampled_from(("", "_consumer", "_at_place_of_employment")),
    condition=st.sampled_from(NAME_CONDITIONS),
)
def test_name_modal_pole_reverses_with_not(
    subject: str, modal: str, verb: str, tail: str, condition: str
):
    """Invariants of a name's modal.

    "may", "can", "must" or "shall" before a verb of conduct grants or
    obliges, so it reads as positive; "not" after it, or "cannot", prohibits,
    so it reads as negative.  A condition clause never changes either.
    """

    assert _name_pole(f"{subject}_{modal}_{verb}{tail}{condition}") is False
    assert _name_pole(f"{subject}_{modal}_not_{verb}{tail}{condition}") is True
    assert _name_pole(f"{subject}_cannot_{verb}{tail}{condition}") is True


@pytest.mark.parametrize(
    "verb",
    [
        "contact",
        "call",
        "visit",
        "email",
        "sue",
        "deny",
        "terminate",
        "disqualify",
        "exceed",
        "affect",
        "constitute",
        "include",
        "waive",
    ],
)
@pytest.mark.parametrize("negated", [False, True])
def test_name_modal_pole_agrees_with_the_clause_reading(verb: str, negated: bool):
    """Differential: a "may" name and its clause read one direction.

    Invariant: the clause "A debt collector may [not] <verb> the consumer if
    the consumer objects." and the name ``debt_collector_may_[not_]<verb>_
    consumer`` agree.  A permission enables and its name reads as positive; a
    prohibition excludes and its name reads as negative; a verb the clause
    reading gives no direction ("may deny", "may not exceed") leaves the name
    with no pole.
    """

    negation = " not" if negated else ""
    clause = (
        f"A debt collector may{negation} {verb} the consumer if the consumer objects."
    )
    name = f"debt_collector_may{negation.replace(' ', '_')}_{verb}_consumer"
    expected = {"enable": False, "exclude": True}.get(_requirement(clause))

    assert _name_pole(name) is expected


@pytest.mark.parametrize(
    ("verb", "participle"),
    [
        ("contact", "contacted"),
        ("notify", "notified"),
        ("accept", "accepted"),
        ("use", "used"),
        ("prorate", "prorated"),
        ("restore", "restored"),
        ("deny", "denied"),
        ("levy", "levied"),
        ("close", "closed"),
        ("terminate", "terminated"),
        ("impose", "imposed"),
        ("include", "included"),
        ("assess", "assessed"),
        ("limit", "limited"),
        ("exceed", "exceeded"),
        ("expel", "expelled"),
        ("cancel", "canceled"),
        ("cancel", "cancelled"),
        ("withhold", "withheld"),
        ("forgive", "forgiven"),
        ("consider", "considered"),
    ],
)
@pytest.mark.parametrize(
    "modal", ["may", "must", "shall", "may_not", "must_not", "cannot"]
)
def test_a_passive_name_reads_as_its_active_form(
    verb: str, participle: str, modal: str
):
    """Metamorphic: voice never changes a name's pole.

    Invariant: ``case_<modal>_be_<participle>`` and ``agency_<modal>_<verb>``
    read one pole, so ``case_must_be_closed`` states no pole, as
    ``agency_must_close`` states none, and ``case_may_be_contacted`` reads as
    positive, as ``agency_may_contact`` does.  A participle that names a
    status (``disqualified``, ``barred``) reads as that status first, so it
    stays out of this table.
    """

    assert _name_pole(f"case_{modal}_be_{participle}") is _name_pole(
        f"agency_{modal}_{verb}"
    )


def _telephone_call_witnesses(formula: str, *, prohibited_above_limit: bool):
    rule = {
        "name": "telephone_call_prohibited",
        "kind": "derived",
        "dtype": "Judgment",
        "versions": [{"formula": formula}],
    }
    above = {
        "name": "above the limit",
        "period": "2026",
        "input": {"calls_placed": 8, "call_limit": 7},
        "output": {"telephone_call_prohibited": prohibited_above_limit},
    }
    at = {
        "name": "at the limit",
        "period": "2026",
        "input": {"calls_placed": 7, "call_limit": 7},
        "output": {"telephone_call_prohibited": not prohibited_above_limit},
    }
    witnesses = completeness_module._toggled_formula_numeric_selectors(
        {"telephone_call_prohibited": rule},
        asserted_by_rule={"telephone_call_prohibited": [above, at]},
        formula_environment={},
    )
    return witnesses, rule


@pytest.mark.parametrize(
    ("formula", "prohibited_above_limit", "matches"),
    [
        ("calls_placed > call_limit", True, True),
        ("calls_placed <= call_limit", False, False),
    ],
)
def test_relational_prohibition_witness_reads_the_output_pole(
    formula: str, prohibited_above_limit: bool, matches: bool
):
    """A relational condition of a prohibition excludes against the output's pole.

    Invariant: in "If the calls placed in the period exceed the call limit, a
    debt collector must not place a telephone call to the consumer.", the
    relation switches the prohibition on, so the relation-active case must
    move ``telephone_call_prohibited`` toward its negative pole.  The faithful
    formula does; the formula that prohibits calls at or under the limit does
    not.
    """

    text = (
        "(1) If the calls placed in the period exceed the call limit, a debt "
        "collector must not place a telephone call to the consumer."
    )
    branch = completeness_module.SourceStructureBranch(
        ("1",), "paragraph", "(1)", text, 0, len(text)
    )
    witnesses, rule = _telephone_call_witnesses(
        formula, prohibited_above_limit=prohibited_above_limit
    )
    relational = [witness for witness in witnesses if witness.relational_transitions]

    assert relational
    assert (
        any(
            completeness_module._numeric_exception_witness_matches_source(
                branch,
                witness,
                extract_numeric_occurrences=EN_NUMERIC_OCCURRENCES,
                rule=rule,
            )
            for witness in relational
        )
        is matches
    )


def _gate_issues(source: str, output: str, formula: str, inputs) -> list[str]:
    document = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": KY_CITATION_PATH}},
        "rules": [
            {
                "name": output,
                "kind": "derived",
                "entity": "TaxUnit",
                "dtype": "Judgment",
                "period": "Year",
                "source": "KRS 141.020",
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": KY_CITATION_PATH,
                                    "excerpt": source.rstrip("."),
                                },
                            }
                        ]
                    }
                },
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            }
        ],
        "inputs": [
            {
                "name": name,
                "entity": "TaxUnit",
                "dtype": "Boolean",
                "period": "Year",
                "description": description,
            }
            for name, description in inputs
        ],
    }
    result = completeness_module.analyze_complete_source_unit(
        yaml.safe_dump(document, sort_keys=False),
        source,
        corpus_citation_path=KY_CITATION_PATH,
        test_cases=[],
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCES,
        extract_numeric_grounding_occurrences=EN_NUMERIC_GROUNDING_OCCURRENCES,
        extract_named_scalars=validator_pipeline_module.extract_named_scalar_occurrences,
        numeric_value_is_grounded=validator_pipeline_module.numeric_value_is_grounded,
    )
    return [
        str(issue)
        for issue in result.issues
        if "source-explicit-conditions" in str(issue)
    ]


OTHERWISE_ELIGIBLE = ("otherwise_eligible", "The person is otherwise eligible.")
DECOMPOSED_GATES = (
    "otherwise_eligible and (not person_is_resident or not person_is_citizen)",
    (
        OTHERWISE_ELIGIBLE,
        ("person_is_resident", "The person is a resident."),
        ("person_is_citizen", "The person is a citizen."),
    ),
)
AGGREGATE_GATE = (
    "otherwise_eligible and not residency_and_status_requirements_hold",
    (
        OTHERWISE_ELIGIBLE,
        (
            "residency_and_status_requirements_hold",
            "The person's residency and status requirements hold.",
        ),
    ),
)


@pytest.mark.parametrize("cue", ["unless", "except when", "except if"])
def test_prohibition_reversal_gates_accept_de_morgan_alternatives(cue: str):
    """A prohibition lifted by joint conditions takes them as De Morgan gates.

    Invariant: in "A person who is otherwise eligible must not claim the
    credit unless that person is a resident and that person is a citizen.",
    either failed condition keeps the prohibition, so a ``_prohibited``
    output that reads each fact separately
    (``otherwise_eligible and (not person_is_resident or not
    person_is_citizen)``) covers both gates, and one that reads an aggregate
    status input does not.  "except when" and "except if" read as "unless".
    """

    source = (
        "A person who is otherwise eligible must not claim the credit "
        f"{cue} that person is a resident and that person is a citizen."
    )

    assert not _gate_issues(source, "credit_claim_prohibited", *DECOMPOSED_GATES)
    assert _gate_issues(source, "credit_claim_prohibited", *AGGREGATE_GATE)


@pytest.mark.parametrize("cue", ["except when", "except if"])
def test_ineligibility_except_when_gates_read_like_unless(cue: str):
    """An "except when" exception to an ineligibility states its gates.

    Invariant: "No person who is otherwise eligible shall be eligible except
    when that person is a resident and that person is a citizen." states the
    same two gates as its "unless" form, so the check accepts the decomposed
    De Morgan formula and rejects an aggregate status input.
    """

    source = (
        "No person who is otherwise eligible shall be eligible "
        f"{cue} that person is a resident and that person is a citizen."
    )

    assert len(completeness_module._source_conjunctive_fact_gates(source)) == 2
    assert not _gate_issues(source, "person_ineligible", *DECOMPOSED_GATES)
    assert _gate_issues(source, "person_ineligible", *AGGREGATE_GATE)


@pytest.mark.parametrize(
    ("matcher", "unit", "count"),
    [
        (
            lambda text: completeness_module._source_modal_effect_matches(text)[0],
            "a debt collector may not communicate, ",
            1,
        ),
        (
            completeness_module._source_lifted_prohibition_matches,
            "the prohibition in paragraph (b) does not apply; ",
            1,
        ),
        (
            completeness_module._source_lifted_prohibition_matches,
            "the prohibitions and ",
            0,
        ),
        (
            completeness_module._source_applied_prohibition_matches,
            "the prohibitions and ",
            0,
        ),
    ],
    ids=[
        "modal-subjects",
        "lifted-prohibitions",
        "lifted-without-boundaries",
        "applied-without-boundaries",
    ],
)
def test_clause_prefix_scans_grow_linearly(matcher, unit: str, count: int):
    """Each match reads a fixed window back to its clause boundary.

    Invariant: finding the subject of every "may not" and the words before
    every lifted prohibition takes work linear in the clause's length.  A
    scan from the start of the text to each match grows with the square of
    the number of matches: about 97 times the work for 10 times the input.
    A clause with no boundary at all ("the prohibitions and " repeated) scans
    a bounded window after each prohibition noun, not the rest of the text.
    """

    def work(repeats: int):
        text = unit * repeats

        def check(matches) -> None:
            assert len(matches) == count * repeats

        return lowest_work(
            lambda: matcher(text), check, setup=completeness_module.re.purge
        )

    assert_linear_growth(work(250), work(2_500), input_ratio=10)


def test_requirement_of_a_clause_without_boundaries_grows_linearly():
    """Reading a prohibition's direction stays linear in the clause's length.

    Invariant: a clause that runs thousands of characters with no ".", ";" or
    ":" ("A debt collector must not communicate with the prohibitions and ...
    the consumer if the consumer objects.") still reads ``exclude``, with work
    linear in its length.
    """

    def work(repeats: int):
        clause = (
            "A debt collector must not communicate with "
            + "the prohibitions and " * repeats
            + "the consumer if the consumer objects."
        )

        def check(requirement: str) -> None:
            assert requirement == "exclude"

        return lowest_work(
            lambda: completeness_module._source_exception_effect_requirement(clause),
            check,
            setup=completeness_module.re.purge,
        )

    assert_linear_growth(work(100), work(1_000), input_ratio=10)
