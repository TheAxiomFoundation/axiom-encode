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

import json
import re

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
    # Round 22: before ``בהתאם ל``, a linker other than ``under``, ``ש`` opens
    # a content clause, so these two refuse; intended stricter, their English
    # accepts (tests/test_source_completeness_hebrew_round22.py).
    ("(א) הטענה שהוגשה בהתאם לסעיף 2 אינה נכונה.", False),
    ("(א) הטענה שנמסרה לפי סעיף 2 אינה נכונה.", True),
    ("(א) הדיווחים שהוגשו בהתאם לסעיף 2 אינם נכונים.", False),
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


# --- Review r9's executed rows, with matched positive controls ------------------

# (id, Hebrew source, English source, verdict, Hebrew positive, English
# positive). Both languages take ``verdict``; both positives accept. Built by
# hand from the targets: r9_replay.py (main session), findings-repro.py,
# extra-spelling-root.py and disclaimer-repro.py (review r9).
UNAFFECTED = (
    "(a) Section 2 remains unaffected and the payment is calculated in "
    "accordance with section 2."
)
EFFECTIVE = UNAFFECTED.replace("unaffected", "effective")
RATHER_IMPOSED = (
    "(a) The amount is determined rather than being imposed in accordance "
    "with section 2."
)
AND_IMPOSED = "(a) The amount is determined and imposed in accordance with section 2."
RATHER_ARISING = (
    "(a) The rights are calculated rather than arising in accordance with section 2."
)
AND_ARISING = "(a) The rights are calculated and arise in accordance with section 2."
ROUND_21_REPRODUCERS = (
    (
        "f1-full-spelling",
        "(א) סעיף 2 ייוותר בעינו והתשלום מחושב בהתאם לסעיף 2.",
        UNAFFECTED,
        False,
        "(א) סעיף 2 ייוותר בתוקפו והתשלום מחושב בהתאם לסעיף 2.",
        EFFECTIVE,
    ),
    (
        "f1-full-spelling-plural",
        "(א) סעיפים 2 ו־3 ייוותרו בעינם והתשלום מחושב בהתאם לסעיף 2.",
        "(a) Sections 2 and 3 remain unaffected and the payment is calculated "
        "in accordance with section 2.",
        False,
        "(א) סעיפים 2 ו־3 ייוותרו בתוקפם והתשלום מחושב בהתאם לסעיף 2.",
        "(a) Sections 2 and 3 remain effective and the payment is calculated "
        "in accordance with section 2.",
    ),
    (
        "f1-generated-spelling",
        "(א) סעיף 2 יוותר בעינו והתשלום מחושב בהתאם לסעיף 2.",
        UNAFFECTED,
        False,
        "(א) סעיף 2 יוותר בתוקפו והתשלום מחושב בהתאם לסעיף 2.",
        EFFECTIVE,
    ),
    (
        "f1-one-citation",
        "(א) התשלום מחושב בהתאם לסעיף 2 אשר ייוותר בעינו.",
        "(a) The payment is calculated in accordance with section 2, which "
        "remains unaffected.",
        False,
        "(א) התשלום מחושב בהתאם לסעיף 2 אשר ייוותר בתוקפו.",
        "(a) The payment is calculated in accordance with section 2, which "
        "remains effective.",
    ),
    (
        "f2-plene-feminine-plural",
        "(א) הזכויות יחושבו במקום שתקומנה בהתאם לסעיף 2.",
        RATHER_ARISING,
        False,
        "(א) הזכויות יחושבו ותקומנה בהתאם לסעיף 2.",
        AND_ARISING,
    ),
    (
        "f2-generated-spelling",
        "(א) הזכויות יחושבו במקום שתקמנה בהתאם לסעיף 2.",
        RATHER_ARISING,
        False,
        "(א) הזכויות יחושבו ותקמנה בהתאם לסעיף 2.",
        AND_ARISING,
    ),
    (
        "f3-plene-future",
        "(א) הסכום ייקבע במקום שיחוייב בהתאם לסעיף 2.",
        RATHER_IMPOSED,
        False,
        "(א) הסכום ייקבע ויחוייב בהתאם לסעיף 2.",
        AND_IMPOSED,
    ),
    (
        "f3-generated-future",
        "(א) הסכום ייקבע במקום שיחויב בהתאם לסעיף 2.",
        RATHER_IMPOSED,
        False,
        "(א) הסכום ייקבע ויחויב בהתאם לסעיף 2.",
        AND_IMPOSED,
    ),
    (
        "f3-plene-feminine",
        "(א) המס ייקבע במקום שתחוייב בהתאם לסעיף 2.",
        "(a) The tax is determined rather than being imposed in accordance "
        "with section 2.",
        False,
        "(א) המס ייקבע ותחוייב בהתאם לסעיף 2.",
        "(a) The tax is determined and imposed in accordance with section 2.",
    ),
    (
        "f3-plene-plural",
        "(א) הסכומים ייקבעו במקום שיחוייבו בהתאם לסעיף 2.",
        "(a) The amounts are determined rather than being imposed in "
        "accordance with section 2.",
        False,
        "(א) הסכומים ייקבעו ויחוייבו בהתאם לסעיף 2.",
        "(a) The amounts are determined and imposed in accordance with section 2.",
    ),
    *(
        (
            f"f4-{index}",
            hebrew_source,
            english_source,
            verdict,
            hebrew_source.replace(" אינה נכונה", " נכונה")
            .replace(" אינם נכונים", " נכונים")
            .replace(" אינן נכונות", " נכונות")
            .replace(" אינו נכון", " נכון"),
            english_source.replace("false", "true").replace("incorrect", "correct"),
        )
        for index, (hebrew_source, english_source, verdict) in enumerate(RELATIVE_TABLE)
    ),
)


@pytest.mark.parametrize(
    (
        "hebrew_source",
        "english_source",
        "verdict",
        "hebrew_positive",
        "english_positive",
    ),
    [row[1:] for row in ROUND_21_REPRODUCERS],
    ids=[row[0] for row in ROUND_21_REPRODUCERS],
)
def test_round_21_reproducers_and_their_matched_positive_controls(
    hebrew_source, english_source, verdict, hebrew_positive, english_positive
):
    assert _pair(hebrew_source, english_source) == (verdict, verdict)
    assert _pair(hebrew_positive, english_positive) == (True, True)


# Finding 5 (review r9 and the main session): English surnames that are German
# negators keep main's verdict, as the neutral ``Lee`` does.
@pytest.mark.parametrize(
    "name",
    (
        "Nie",
        "Ms Nie",
        "Kein",
        "Ms Kein",
        "Keine",
        "Keiner",
        "Weder",
        "Statt",
        "Anstatt",
        "Ausser",
        "Nicht",
        "Niemals",
        "Ohne",
        "Lee",
        "Ms Lee",
    ),
)
def test_round_21_english_surnames_keep_mains_verdict(name):
    for frame in (
        "(a) Tax is computed for {} in accordance with section 2.",
        "(a) The amount is calculated for {} in accordance with section 2.",
    ):
        assert _english_accepted("a", ENGLISH_REASON, frame.format(name))


# --- Findings 1–3: the spellings, closed from the snapshot ----------------------


def test_full_spelling_disclaimers_are_generated_and_the_homographs_are_not():
    assert {"ייוותר", "ייוותרו"} <= set(negators.DISCLAIMER_VERBS)
    assert not {"יותר", "יותרו"} & set(negators.DISCLAIMER_VERBS)


PLENE_CHARGE = (
    "חוייב",
    "חוייבה",
    "חוייבו",
    "מחוייב",
    "מחוייבת",
    "מחוייבים",
    "מחוייבות",
    "יחוייב",
    "תחוייב",
    "יחוייבו",
    "תחוייבנה",
)


def test_every_charge_slot_has_its_plene_spelling():
    family = negators.VERB_PARADIGMS["חויב"]
    assert set(PLENE_CHARGE) <= set(family.forms)
    for form in ("יחוייב", "תחוייב", "יחוייבו", "תחוייבנה"):
        assert form in negators.COMPUTATION_RELATIVE_FUTURES
    for relative in ("שיחוייב", "שתחוייב", "שיחוייבו", "שתחוייבנה"):
        source = f"(א) הסכום ייקבע במקום {relative} בהתאם לסעיף 2."
        assert not _accepted("a", HEBREW_REASON, source), relative


def _closure() -> dict:
    from scripts import generate_hebrew_paradigm_audit as audit

    fixture = json.loads(audit.FIXTURE.read_text())
    return fixture["round21"]


def _generated_forms() -> set[str]:
    return {
        form
        for registry in (
            negators.NOUN_PARADIGMS,
            negators.ADJECTIVE_PARADIGMS,
            negators.VERB_PARADIGMS,
        )
        for family in registry.values()
        for form in family.forms
    }


def test_every_holam_alternate_is_added_or_refused_as_recorded():
    forms = _generated_forms()
    rows = _closure()["holam_alternates"]
    assert rows
    for row in rows:
        family = negators.VERB_PARADIGMS[row["family"]]
        if row["decision"] == "added":
            assert row["form"] in family.spellings(row["slot"]), row
        else:
            assert row["decision"] == "refused" and row["reason"], row
            assert row["form"] not in forms, row
    added = {row["form"] for row in rows if row["decision"] == "added"}
    assert {"תקומנה", "יעמודו", "תעמדנה", "יכפורו", "תכפרנה", "ייוותר"} <= added


def test_listed_defective_spellings_stay_out():
    forms = _generated_forms()
    defective = _closure()["defective_exclusions"]
    assert set(defective) == {
        "יקבע",
        "יקבעו",
        "תקבע",
        "יחשב",
        "תחשב",
        "מחשב",
        "מחשבת",
        "מחשבים",
        "חשב",
        "עמדת",
        "עמדות",
        "דווח",
        "דווחו",
        "אמרה",
        "עבוד",
        "יחוב",
    }
    assert not forms & set(defective)


def test_every_snapshot_neighbour_is_generated_or_named():
    """The pinned inventory: snapshot words one ``ו`` or ``י`` from a generated form."""

    forms = _generated_forms()
    rows = _closure()["neighbours"]
    assert len(rows) >= 200
    named = set()
    for row in rows:
        if row["class"] == "generated":
            assert row["word"] in forms, row
        else:
            assert row["class"] in {"homograph", "other lexeme", "defective active"}
            assert row["reason"], row
            assert row["word"] not in forms, row
            named.add(row["word"])
        assert row["near"] and all(near in forms for near in row["near"]), row
    # No generated form holds a name in any exclusion table.
    closure = _closure()
    refused = {
        row["form"]
        for row in closure["holam_alternates"]
        if row["decision"] == "refused"
    }
    assert not forms & (named | refused | set(closure["defective_exclusions"]))


def test_a_new_neighbour_without_a_class_fails_the_check():
    from scripts import generate_hebrew_paradigm_audit as audit

    rows = audit.classify_neighbours(
        [
            {"word": "קקקק", "count": 1, "near": ["חויב"]},
            {"word": "יחוייב", "count": 9, "near": ["יחויב"]},
            {"word": "יקבע", "count": 1, "near": ["ייקבע"]},
        ]
    )
    assert [row["class"] for row in rows] == [
        "unclassified",
        "generated",
        "defective active",
    ]
    assert audit.spelling_closure(None, _closure()) == _closure()


def test_compact_alternation_refuses_an_empty_language():
    with pytest.raises(ValueError):
        negators.compact_alternation([])
    with pytest.raises(ValueError):
        negators.compact_alternation([""])
    assert re.fullmatch(negators.compact_alternation(["אב", "אג"]), "אג")
