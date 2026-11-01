"""Round 21 of #1882: relatives are no claims, closed spellings, small repairs.

Review r9 (GPT-6.1 Sol, head 82e71841a) found three spellings of listed words
the generator missed (findings 1–3) and attributive relatives that the claim
reader took for denied claims (finding 4). Every expectation here is written
by hand from the targets the main session set, not derived from the
generator: generator-derived expectations alone missed findings 1–4.

A row is ``(Hebrew source, faithful English source, Hebrew verdict, English
verdict)``. A negative row also names its matched positive control: the same
English with the negative word removed or turned positive, which accepts.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness import hebrew_negators as negators
from axiom_encode.harness import hebrew_source_text as hebrew
from tests.test_source_completeness_hebrew import _accepted, _english_accepted

HEBREW_REASON = "חסר סעיף 2"
ENGLISH_REASON = "Missing section 2."
FALSE_CLAIM = (
    "(a) It is false that the payment is computed in accordance with section 2."
)
TRUE_CLAIM = "(a) It is true that the payment is computed in accordance with section 2."


def _pair(hebrew_source: str, english_source: str) -> tuple[bool, bool]:
    return (
        _accepted("a", HEBREW_REASON, hebrew_source),
        _english_accepted("a", ENGLISH_REASON, english_source),
    )


# --- Finding 4: a claim noun's own relative clause is no claim ------------------

# The main session's table: attributive relatives accept in both languages,
# content clauses refuse in both.
RELATIVE_TABLE = (
    (
        "(א) דיווחיו שהוגשו לפי סעיף 2 אינם נכונים.",
        "(a) His reports filed under section 2 are incorrect.",
        True,
    ),
    (
        "(א) דיווחי הנישום שהוגשו לפי סעיף 2 אינם נכונים.",
        "(a) The taxpayer's reports filed under section 2 are incorrect.",
        True,
    ),
    (
        "(א) הדיווחים שהוגשו לפי סעיף 2 אינם נכונים.",
        "(a) The reports filed under section 2 are incorrect.",
        True,
    ),
    (
        "(א) ההנחות שניתנו לפי סעיף 2 אינן נכונות.",
        "(a) The discounts granted under section 2 are incorrect.",
        True,
    ),
    (
        "(א) ההנחה שניתנה לפי סעיף 2 אינה נכונה.",
        "(a) The discount granted under section 2 is incorrect.",
        True,
    ),
    (
        "(א) הטענה שהועלתה לפי סעיף 2 אינה נכונה.",
        "(a) The claim raised under section 2 is incorrect.",
        True,
    ),
    (
        "(א) דיווחיו שהוגשו על ידו לפי סעיף 2 אינם נכונים.",
        "(a) His reports filed by him under section 2 are incorrect.",
        True,
    ),
    (
        "(א) ההנחות שניתנו לנישום לפי סעיף 2 אינן נכונות.",
        "(a) The discounts granted to the taxpayer under section 2 are incorrect.",
        True,
    ),
    (
        "(א) הדוח שהוגש לפי סעיף 2 אינו נכון.",
        "(a) The report filed under section 2 is incorrect.",
        True,
    ),
    ("(א) הטענה שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.", FALSE_CLAIM, False),
    ("(א) טענתו שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.", FALSE_CLAIM, False),
    (
        "(א) הטענה שהזכות קיימת בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the right exists in accordance with section 2.",
        False,
    ),
    (
        "(א) הטענה שחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that it was computed in accordance with section 2.",
        False,
    ),
    (
        "(א) הטענה כי הוגשו בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that they were filed in accordance with section 2.",
        False,
    ),
)


@pytest.mark.parametrize(("hebrew_source", "english_source", "verdict"), RELATIVE_TABLE)
def test_relatives_are_no_claims_and_content_clauses_stay_claims(
    hebrew_source, english_source, verdict
):
    assert _pair(hebrew_source, english_source) == (verdict, verdict)
    if not verdict:
        # The matched positive control: the same English, turned true.
        assert _english_accepted(
            "a", ENGLISH_REASON, english_source.replace("false", "true")
        )
        assert _accepted(
            "a", HEBREW_REASON, hebrew_source.replace(" אינה נכונה", " נכונה")
        )


# Hand-written controls of the rule as built. ``under`` links (``לפי``) take
# the one-word relative; English refuses no claim before them either.
RELATIVE_CONTROLS = (
    # A masculine singular verb after a feminine noun cannot be its relative:
    # ``הוסכם`` is impersonal, "it was agreed that".
    ("(א) הטענה שהוסכם בהתאם לסעיף 2 אינה נכונה.", False),
    # A relative that a content clause follows still holds a claim.
    ("(א) הטענה שהועלתה אתמול כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.", False),
    # ``כופר בטענה`` refuses whether a relative or a content clause follows.
    ("(א) הפקיד כופר בטענה שהועלתה בהתאם לסעיף 2.", False),
    ("(א) הפקיד כופר בטענה שהתשלום מחושב בהתאם לסעיף 2.", False),
    ("(א) ההנחות שיינתנו לפי סעיף 2 אינן נכונות.", True),
    ("(א) הטענה שיחושב בהתאם לסעיף 2 אינה נכונה.", False),
    ("(א) הטענה שהוגשה בהתאם לסעיף 2 אינה נכונה.", True),
    ("(א) הטענה שנמסרה לפי סעיף 2 אינה נכונה.", True),
    ("(א) הדיווחים שהוגשו בהתאם לסעיף 2 אינם נכונים.", True),
    # Agreement: a feminine singular verb is no relative of a plural noun.
    ("(א) הדיווחים שהוגשה בהתאם לסעיף 2 אינם נכונים.", False),
    # A computation verb keeps the content reading, even one word from the link.
    ("(א) הטענה שנקבע לפי סעיף 2 אינה נכונה.", False),
    ("(א) הטענה ש־הועלתה לפי סעיף 2 אינה נכונה.", True),
    ("(א) טענותיו שהועלו לפי סעיף 2 אינן נכונות.", True),
    ("(א) טענות הנישום שהועלו לפי סעיף 2 אינן נכונות.", True),
    ("(א) הדיווח שהתקבל לפי סעיף 2 אינו נכון.", True),
    ("(א) הדיווח שניתן לפי סעיף 2 אינו נכון.", True),
    # Modal ``ניתן ל`` ("it is possible to") opens a content clause.
    ("(א) הדיווח שניתן לחשב בהתאם לסעיף 2 אינו נכון.", False),
)


@pytest.mark.parametrize(("source", "verdict"), RELATIVE_CONTROLS)
def test_relative_rule_controls(source, verdict):
    assert _accepted("a", HEBREW_REASON, source) is verdict


# Intended stricter: a relative with no listed verb and two or more words
# before the link keeps the claim reading and refuses, while its English
# accepts. The pinned snapshot holds nine such relatives after a claim noun
# (``הנחות שיינתנו למי ששילם``, ``דיווח חודשי מקוון כפי שיורה המוסד``), and
# none of its 45 claim matches has a postposed denial in its clause, so none
# moves a snapshot verdict (.review-scratch/round21/REPORT.md).
INTENDED_STRICTER_RELATIVES = (
    (
        "(א) ההנחות שיינתנו לנישום לפי סעיף 2 אינן נכונות.",
        "(a) The discounts to be granted to the taxpayer under section 2 are incorrect.",
    ),
    (
        "(א) הטענה שנמסרה בהתאם לסעיף 2 אינה נכונה.",
        "(a) The claim delivered in accordance with section 2 is incorrect.",
    ),
    (
        "(א) הדיווחים שהנישום הגיש לפי סעיף 2 אינם נכונים.",
        "(a) The reports the taxpayer filed under section 2 are incorrect.",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source"), INTENDED_STRICTER_RELATIVES
)
def test_unlisted_long_relatives_are_intended_stricter(hebrew_source, english_source):
    assert _pair(hebrew_source, english_source) == (False, True)


def test_attributive_verbs_agree_and_exclude_impersonals_and_homographs():
    assert "הוגשו" in negators.attributive_forms("masculine plural")
    assert "הוגשו" in negators.attributive_forms("feminine plural")
    assert "הועלתה" in negators.attributive_forms("feminine singular")
    assert "הועלתה" not in negators.attributive_forms("masculine singular")
    for agreement in negators._AGREEMENT_SLOTS:
        forms = set(negators.attributive_forms(agreement))
        assert not forms & set(negators.ATTRIBUTIVE_EXCLUSIONS), agreement
        computation = {
            form
            for paradigm in negators.COMPUTATION_VERB_PARADIGMS
            for form in paradigm.forms
        }
        assert not forms & computation, agreement
    for form in ("ניתן", "נטען", "הוסכם", "מוסכם", "מוסר", "מצוין", "מצויין"):
        assert form in negators.ATTRIBUTIVE_EXCLUSIONS


def test_the_claim_reader_with_relatives_is_bounded():
    """A long run of words after a claim noun and ``ש`` costs linear time."""

    import time

    words = " ".join(["מילה"] * 4000)
    for text in (
        f"הטענה שהוגשה {words} כי",
        f"הטענה ש{words} לפי סעיף 2",
        "הטענה " + "שהוגשה " * 4000,
    ):
        started = time.process_time()
        list(hebrew._CLAIM_SUBJECT.finditer(text))
        assert time.process_time() - started < 1.5
