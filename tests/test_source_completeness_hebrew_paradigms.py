"""Round 20: close each listed Hebrew family by its grammatical paradigm.

The examples preserve the review's exact Hebrew wording. Claim denials use
the canonical English order which actually refuses; ``The claim ... is false``
has a different English reading and is not their canonical control.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import hebrew_negators as negators
from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness.hebrew_prefixes import PREFIX_JOINT_CHARS
from scripts import generate_hebrew_paradigm_audit as audit_generator
from tests.test_source_completeness_hebrew import _accepted, _english_accepted

_REASON = "חסר סעיף 2"
_ENGLISH_REASON = "Missing section 2."
_DENIAL = (
    "(a) The official denies the claim that the payment is calculated in accordance "
    "with section 2."
)
_AFFIRMATION = _DENIAL.replace("denies", "affirms")
_FALSE = "(a) It is false that the payment is computed in accordance with section 2."
_TRUE = _FALSE.replace("false", "true")
_UNAFFECTED = (
    "(a) The provisions of section 2 remain unaffected and the payment is "
    "calculated in accordance with section 2."
)
_EFFECTIVE = _UNAFFECTED.replace("unaffected", "effective")
_WITHOUT = "(a) The payment shall be without calculation in accordance with section 2."
_WITH = _WITHOUT.replace("without", "with")
_DISCLAIMS = "(a) The official disclaims calculation in accordance with section 2."
_AFFIRMS = _DISCLAIMS.replace("disclaims", "affirms")

# Every source omission in the r8 findings and r8_scope.py, plus the already
# refusing siblings used in those replays. Each negative has its own matched
# positive in both languages; no English acceptance is used to excuse a miss.
# (id, negative Hebrew, negative English, positive Hebrew, positive English)
ROUND_20_REPRODUCERS = (
    *(
        (
            f"computation-{noun}",
            f"(א) התשלום יהיה נטול {noun} בהתאם לסעיף 2.",
            _WITHOUT,
            f"(א) התשלום יהיה עם {noun} בהתאם לסעיף 2.",
            _WITH,
        )
        for noun in (
            "יישומים",
            "יישום",
            "יישומו",
            "יישומה",
            "יישומם",
            "חישוב",
            "חישוביו",
            "ישום",
            "ישומים",
            "קביעות",
            "קביעותיו",
            "חיוביו",
            "חיובים",
            "הטלותיו",
            "חישובים",
            "תחולות",
            "צמיחות",
            "השתות",
            "הטלות",
            "החלות",
            "ה־חישוב",
            "כל־חישוב",
        )
    ),
    (
        "computation-relative-feminine-future",
        "(א) ההכנסות נקבעות במקום שתחושבנה בהתאם לסעיף 2.",
        "(a) The incomes are fixed rather than being calculated in accordance with section 2.",
        "(א) ההכנסות נקבעות ותחושבנה בהתאם לסעיף 2.",
        "(a) The incomes are fixed and calculated in accordance with section 2.",
    ),
    *(
        (
            f"claim-{head}-{claim}",
            f"(א) הפקיד {head} ב{claim} שהתשלום מחושב בהתאם לסעיף 2.",
            _DENIAL,
            f"(א) הפקיד מאשר את {claim} שהתשלום מחושב בהתאם לסעיף 2.",
            _AFFIRMATION,
        )
        for head, claim in (
            ("כופר", "טענתו"),
            ("כופר", "טענה"),
            ("כופר", "טענת הנישום"),
            ("כופר", "טענת John Smith"),
            ("כופר", "טענותיו"),
            ("כופר", "הנחות"),
            ("כופר", "גרסאות"),
            ("כופר", "גירסה"),
            ("כפר", "טענה"),
            ("יכפור", "טענה"),
        )
    ),
    *(
        (
            f"postposed-{claim}",
            f"(א) {claim} שהתשלום מחושב בהתאם לסעיף 2 {predicate}.",
            _FALSE,
            f"(א) {claim} שהתשלום מחושב בהתאם לסעיף 2 {positive}.",
            _TRUE,
        )
        for claim, predicate, positive in (
            ("הטענה", "אינה נכונה", "נכונה"),
            ("טענתו", "אינה נכונה", "נכונה"),
            ("טענותיו", "אינן נכונות", "נכונות"),
            ("הטענות", "אינן נכונות", "נכונות"),
            ("הנחתו", "שגויה", "נכונה"),
            ("הגירסה", "שגויה", "נכונה"),
            ("טענת הנישום", "אינה נכונה", "נכונה"),
            ("טענת John Smith", "אינה נכונה", "נכונה"),
        )
    ),
    *(
        (
            f"disclaimer-{subject}-{predicate}",
            f"(א) {subject} {predicate} והתשלום מחושב בהתאם לסעיף 2.",
            _UNAFFECTED,
            f"(א) {subject} בתוקף והתשלום מחושב בהתאם לסעיף 2.",
            _EFFECTIVE,
        )
        for subject, predicate in (
            ("הוראות סעיף 2", "תעמודנה בעינן"),
            ("הוראות סעיף 2", "תיוותרנה בעינן"),
            ("הוראות סעיף 2", "עומדות בעינן"),
            ("הוראות סעיף 2", "עמדו בעינן"),
            ("סעיף 2", "עמד בעינו"),
            ("סעיף 2", "נשאר בעינו"),
            ("סעיף 2", "יישאר בעינו"),
            ("סעיף 2", "ימשיך לעמוד בעינו"),
            ("סעיף 2", "נותר־בעינו"),
        )
    ),
    *(
        (
            f"phrase-{phrase}",
            f"(א) הפקיד {phrase} בהתאם לסעיף 2.",
            _DISCLAIMS,
            "(א) הפקיד מאשר חישוב בהתאם לסעיף 2.",
            _AFFIRMS,
        )
        for phrase in ("מתנער מ־חישוב", "מתנער מחישוב")
    ),
    *(
        (
            f"false-{phrase}",
            f"(א) {phrase} מחושב בהתאם לסעיף 2.",
            _FALSE,
            "(א) אמת שהתשלום מחושב בהתאם לסעיף 2.",
            _TRUE,
        )
        for phrase in ("שקר ש־התשלום", "שקר כי התשלום", "שקר שהתשלום")
    ),
    (
        "phrase-excluded-from",
        "(א) החישוב הוצא מ־כלל התשלום בהתאם לסעיף 2.",
        "(a) The calculation was excluded from the payment in accordance with section 2.",
        "(א) החישוב נכלל בתשלום בהתאם לסעיף 2.",
        "(a) The calculation was included in the payment in accordance with section 2.",
    ),
    (
        "phrase-governed-infinitive-excluded-from",
        "(א) יש להוציא מכלל החישוב בהתאם לסעיף 2.",
        "(a) The calculation must be excluded from the payment in accordance with section 2.",
        "(א) יש לכלול את החישוב בהתאם לסעיף 2.",
        "(a) The calculation must be included in the payment in accordance with section 2.",
    ),
)


@pytest.mark.parametrize(
    ("row_id", "source", "english", "positive", "english_positive"),
    ROUND_20_REPRODUCERS,
    ids=[row[0] for row in ROUND_20_REPRODUCERS],
)
def test_round_20_reproducers_and_their_matched_positive_controls(
    row_id, source, english, positive, english_positive
):
    assert not _english_accepted("a", _ENGLISH_REASON, english), row_id
    assert not _accepted("a", _REASON, source), row_id
    assert _english_accepted("a", _ENGLISH_REASON, english_positive), row_id
    assert _accepted("a", _REASON, positive), row_id


@pytest.mark.parametrize(
    "source",
    (
        "(א) הוטל עליו כופר כסף בהתאם לסעיף 2.",
        "(א) במקום שיקבע השר לפי סעיף 2 יחושב התשלום.",
        "(א) שיקום בהתאם לסעיף 2.",
        "(א) שיקומו בהתאם לסעיף 2.",
        "(א) מונע תשלום בהתאם לסעיף 2.",
        "(א) סעיף 2 נשאר בתוקפו והתשלום מחושב בהתאם לסעיף 2.",
        "(א) הדוח שהוגש לפי סעיף 2 אינו נכון.",
        "(א) יש להוציא רישיון לפי סעיף 2.",
    ),
)
def test_paradigm_closure_preserves_the_statutory_homographs(source):
    assert _accepted("a", _REASON, source)


def test_the_exact_instead_of_review_wording_is_labelled_english_sensitive():
    """The literal review English accepts; the faithful rather-than frame refuses."""

    source = "(א) ההכנסות נקבעות במקום שתחושבנה בהתאם לסעיף 2."
    english = (
        "(a) The incomes are fixed instead of being calculated in accordance "
        "with section 2."
    )
    assert not _accepted("a", _REASON, source)
    assert _english_accepted("a", _ENGLISH_REASON, english)


@pytest.mark.parametrize(
    "owner", ("", "הנישום", "בן הנישום", "בן בן הנישום", "John Smith")
)
def test_a_construct_claim_allows_at_most_three_owner_tokens(owner):
    negative = f"(א) הפקיד כופר בטענת {owner} שהתשלום מחושב בהתאם לסעיף 2."
    assert not _accepted("a", _REASON, negative)
    assert _accepted("a", _REASON, negative.replace("כופר ב", "מאשר את "))


@pytest.mark.parametrize(
    "owner",
    (
        "בן בן בן הנישום",
        "הנישום,",
        "הנישום.",
        "הנישום\nהאחר",
        "הנישום\n",
        "הנישום \n האחר",
        "הנישום\u2028האחר",
    ),
)
def test_a_construct_claim_never_scans_a_long_or_separate_owner_clause(owner):
    source = f"(א) הפקיד כופר בטענת {owner} שהתשלום מחושב בהתאם לסעיף 2."
    assert not hebrew._NEGATIONS.search(source), source
    postposed = f"(א) טענת {owner} שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה."
    assert not hebrew._CLAIM_SUBJECT.search(postposed), postposed


def _families():
    return (
        *negators.NOUN_PARADIGMS.values(),
        *negators.VERB_PARADIGMS.values(),
        *negators.ADJECTIVE_PARADIGMS.values(),
    )


def test_generated_noun_and_verb_forms_have_final_letters_only_at_the_end():
    for family in _families():
        for form in family.forms:
            for word in form.split():
                assert not set(word[:-1]) & set("םןךףץ"), (family, form)


def test_every_listed_paradigm_fills_each_slot_or_explains_its_absence():
    for family in negators.NOUN_PARADIGMS.values():
        assert len(family.absolute) == 2, family
        assert len(family.construct) == 2, family
        assert len(family.singular_possessive) == 10, family
        assert len(family.plural_possessive) == 10, family
    for family in negators.VERB_PARADIGMS.values():
        for slot, count in (("past", 3), ("present", 4), ("future", 4)):
            forms = getattr(family, slot)
            assert len(forms) == count or family.missing_slots.get(slot), (family, slot)
        assert family.infinitive or family.missing_slots.get("infinitive"), family
    for family in negators.ADJECTIVE_PARADIGMS.values():
        assert len(family.absolute) == 4, family
        assert len(family.construct) == 4, family
        for slot in ("past", "future", "infinitive"):
            assert dict(family.missing_slots).get(slot), (family, slot)


def test_final_nun_assimilates_before_the_feminine_plural_future_ending():
    assert negators.VERB_PARADIGMS["מיאן"].future[-1] == "תמאנה"
    assert negators.VERB_PARADIGMS["הבדיל"].future[-1] == "תבדלנה"


_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/hebrew_rendering_audit.json").read_text()
)
_PARADIGM_FRAMES = audit_generator.family_frames()
_PREFIX_SAMPLE = ("", "ו", "ש", "וש")
_JOINTS = tuple(PREFIX_JOINT_CHARS)
_REGISTRIES = {
    "noun": negators.NOUN_PARADIGMS,
    "verb": negators.VERB_PARADIGMS,
    "adjective": negators.ADJECTIVE_PARADIGMS,
}
_INFINITIVES = {
    form for family in negators.VERB_PARADIGMS.values() for form in family.infinitive
}


def _forms(row):
    family = _REGISTRIES[row["registry"]][row["family"]]
    return tuple(
        form
        for form in getattr(family, row["slot"])
        if form not in row.get("excluded_forms", {})
    )


def _prefixed(prefix: str, word: str, joint: str) -> str:
    if word in _INFINITIVES:
        word = word[0] + joint + word[1:]
    return joint.join((*prefix, word)) if prefix else word


def _source_pair(row, form, prefix="", joint=""):
    """Expand a generated family slot into its negative and neutral frames."""

    role = row["role"]
    family = _REGISTRIES[row["registry"]][row["family"]]
    head = _prefixed(prefix, form, joint)
    neutral = _prefixed(prefix, "מדויק", joint)
    end = " בהתאם לסעיף 2."
    if role in {"computation", "computational-processing"}:
        noun = form + (" חישובי" if role == "computational-processing" else "")
        head = _prefixed(prefix, "נטול", joint)
        return (
            f"(א) התשלום יהיה {head} {noun}{end}",
            f"(א) התשלום יהיה {neutral} {noun}{end}",
        )
    if role in {"computational-adjective", "passive-computation"}:
        noun = (
            f"עיבוד {form}"
            if role == "computational-adjective"
            else f"ל{joint}היות {form}"
        )
        head = _prefixed(prefix, "נטול", joint)
        return (
            f"(א) התשלום יהיה {head} {noun}{end}",
            f"(א) התשלום יהיה {neutral} {noun}{end}",
        )
    if role in {"relative-computation", "infinitive-computation"}:
        relative = "ש" + joint if role == "relative-computation" else ""
        form = _prefixed("", form, joint)
        head = _prefixed(prefix, "במקום", joint)
        return (
            f"(א) התשלום נקבע {head} {relative}{form}{end}",
            f"(א) התשלום נקבע {neutral} {relative}{form}{end}",
        )
    if role in {"claim", "postposed-claim"}:
        owner = " הנישום" if form in family.construct else ""
        subject = form if role == "claim" else head
        content = f"{subject}{owner} ש{joint}התשלום מחושב בהתאם לסעיף 2"
        if role == "claim":
            denial = _prefixed(prefix, "כופר", joint)
            return (
                f"(א) הפקיד {denial} ב{joint}{content}.",
                f"(א) הפקיד מאשר את {content}.",
            )
        return f"(א) {content} היא שקר.", f"(א) {content} היא אמת."
    if role == "clause-negator":
        return (
            f"(א) {head} התשלום מחושב{end}",
            f"(א) {neutral} התשלום מחושב{end}",
        )
    if role in {"privative", "construct-complement"}:
        return (
            f"(א) התשלום יהיה {head} חישוב{end}",
            f"(א) התשלום יהיה {neutral} חישוב{end}",
        )
    if role == "phrase":
        tail = "חישוב" if row["family"] == "התנער" else "כלל התשלום"
        return (
            f"(א) הפקיד {head} מ{joint}{tail}{end}",
            f"(א) הפקיד {neutral} מ{joint}{tail}{end}",
        )
    if role == "false-phrase":
        return (
            f"(א) {head} ש{joint}התשלום מחושב{end}",
            f"(א) אמת ש{joint}התשלום מחושב{end}",
        )
    if role == "claim-denial":
        return (
            f"(א) הפקיד {head} ב{joint}טענה ש{joint}התשלום מחושב{end}",
            f"(א) הפקיד מאשר בטענה ש{joint}התשלום מחושב{end}",
        )
    if role == "disclaimer":
        continuation = "ימשיך " if form in family.infinitive else ""
        return (
            f"(א) סעיף 2 {continuation}{head} ב{joint}עינו והתשלום מחושב{end}",
            f"(א) סעיף 2 בתוקף והתשלום מחושב{end}",
        )
    if role == "continued-disclaimer":
        return (
            f"(א) סעיף 2 {head} ל{joint}עמוד ב{joint}עינו והתשלום מחושב{end}",
            f"(א) סעיף 2 בתוקף והתשלום מחושב{end}",
        )
    if role == "absent-predicate":
        return (
            f"(א) החישוב {head} מ{joint}התשלום המחושב{end}",
            f"(א) החישוב {neutral} מ{joint}התשלום המחושב{end}",
        )
    if role == "from-complement":
        return (
            f"(א) התשלום {head} מ{joint}חישוב{end}",
            f"(א) התשלום {neutral} מ{joint}חישוב{end}",
        )
    if role in {"postposed-correct", "postposed-false"}:
        predicate = (
            f"{_prefixed(prefix, 'אינו', joint)} {form}"
            if role == "postposed-correct"
            else head
        )
        claim = f"(א) הטענה ש{joint}התשלום מחושב בהתאם לסעיף 2"
        return f"{claim} {predicate}.", f"{claim} אמת."
    raise AssertionError(role)


def test_the_paradigm_fixture_is_generated_and_keeps_every_reason_form():
    assert _FIXTURE["round20"]["frames"] == _PARADIGM_FRAMES
    assert _FIXTURE["round20"]["reason_baseline"] == audit_generator.BASELINE
    assert [row[0] for row in _FIXTURE["round20"]["reason_verdicts"]] == (
        audit_generator.reason_forms()
    )
    represented = {(row["registry"], row["family"]) for row in _PARADIGM_FRAMES}
    for registry, families in _REGISTRIES.items():
        assert all((registry, family) in represented for family in families), registry


@pytest.mark.parametrize("row", _PARADIGM_FRAMES, ids=lambda row: row["cls"])
def test_every_generated_form_refuses_with_a_neutral_source_control(row):
    """Every form × four ordinary prefixes × every visible joint, source reader."""

    prefixes = (
        negators.ORDINARY_PREFIX_CHAINS
        if row["role"]
        in {
            "disclaimer",
            "continued-disclaimer",
            "postposed-claim",
            "postposed-correct",
            "postposed-false",
        }
        else _PREFIX_SAMPLE
    )
    for form, prefix, joint in itertools.product(_forms(row), prefixes, _JOINTS):
        negative, positive = _source_pair(row, form, prefix, joint)
        assert not hebrew._linked_source_instruments(negative), negative
        assert hebrew._linked_source_instruments(positive), positive


@settings(max_examples=100, deadline=None)
@given(
    row=st.sampled_from(_PARADIGM_FRAMES),
    index=st.integers(min_value=0, max_value=1000),
    prefix=st.sampled_from(negators.ORDINARY_PREFIX_CHAINS),
    joint=st.sampled_from(_JOINTS),
)
def test_generated_paradigms_have_the_same_public_source_verdict(
    row, index, prefix, joint
):
    forms = _forms(row)
    form = forms[index % len(forms)]
    negative, positive = _source_pair(row, form, prefix, joint)
    assert not _accepted("a", _REASON, negative), negative
    assert _accepted("a", _REASON, positive), positive
    assert not _english_accepted("a", _ENGLISH_REASON, row["english"][0]), row
    assert _english_accepted("a", _ENGLISH_REASON, row["english"][1]), row


def _phrase_pair(phrase, prefix, joint):
    head, complement = phrase.split()
    if complement == "מ":
        tail = "מ" + joint + "חישוב"
    elif complement == "מכלל":
        tail = "מ" + joint + "כלל התשלום"
    elif complement == "ש":
        tail = "ש" + joint + "התשלום מחושב"
    else:
        assert complement == "כי", phrase
        tail = "כי התשלום מחושב"
    return tuple(
        f"(א) {_prefixed(prefix, marker, joint)} {tail} בהתאם לסעיף 2."
        for marker in (head, "מדויק")
    )


@pytest.mark.parametrize("phrase", negators.SOURCE_CLAUSE_PHRASES)
def test_every_phrase_internal_joint_and_ordinary_prefix(phrase):
    """Sol's 5,488 products and every phrase added by the closed paradigms."""

    for prefix, joint in itertools.product(negators.ORDINARY_PREFIX_CHAINS, _JOINTS):
        negative, positive = _phrase_pair(phrase, prefix, joint)
        assert not hebrew._linked_source_instruments(negative), negative
        assert hebrew._linked_source_instruments(positive), positive


@pytest.mark.parametrize("state", negators.DISCLAIMER_STATE_FORMS)
def test_every_disclaimer_agrees_with_every_state_suffix(state):
    for form, joint in itertools.product(
        (*negators.DISCLAIMER_VERBS, *negators.DISCLAIMER_INFINITIVES), _JOINTS
    ):
        head = _prefixed("", form, joint)
        visible_state = state[0] + joint + state[1:]
        source = f"(א) סעיף 2 {head} {visible_state} והתשלום מחושב בהתאם לסעיף 2."
        assert not hebrew._linked_source_instruments(source), source
        assert hebrew._linked_source_instruments(
            source.replace(f"{head} {visible_state}", "בתוקף")
        ), source


def test_a_governed_explicit_exclusion_preserves_every_internal_joint():
    for joint in _JOINTS:
        source = f"(א) יש ל{joint}הוציא מ{joint}כלל החישוב בהתאם לסעיף 2."
        assert not hebrew._linked_source_instruments(source), source
        assert hebrew._linked_source_instruments(
            "(א) יש לכלול את החישוב בהתאם לסעיף 2."
        )
    # The source-only override leaves the old issuing governor in reasons.
    assert not hebrew._REASON_NEGATION.search("יש להוציא מכלל חישוב חסר סעיף 2")


@pytest.mark.parametrize(
    "rows",
    [
        _FIXTURE.get("round20", {}).get("reason_verdicts", [])[index : index + 50]
        for index in range(
            0, len(_FIXTURE.get("round20", {}).get("reason_verdicts", [])), 50
        )
    ],
)
def test_every_paradigm_form_keeps_the_reason_verdict_at_0acff3551(rows):
    source = "(א) התשלום מחושב בהתאם לסעיף 2."
    for form, expected in rows:
        actual = [
            hebrew.reason_names_source_bound_dependency(
                reason,
                source,
                corpus_citation_path="il/statute/test-law/section-1",
            )
            for reason in audit_generator.reason_frames(form)
        ]
        assert actual == expected, form
