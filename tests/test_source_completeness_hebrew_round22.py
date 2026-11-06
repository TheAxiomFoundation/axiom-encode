"""Round 22 of #1882: five small closures from review r10.

Review r10 (GPT-6.1 Sol, head 5dcf2bc13) found a Hebrew rule annotation that
reached past the wall (finding 1, in tests/test_source_completeness_non_hebrew_wall.py),
an agreement guess that read denied content as a relative (finding 2),
object-suffixed computation infinitives (finding 3), the noun ``דוחות``
("reports") read as "reject" (finding 4), and binary searches that grew a
linear family by a log factor (finding 5). Every expectation here is written
by hand from the targets the main session set, not derived from the
generator.

A row is ``(id, Hebrew source, faithful English source, Hebrew verdict,
English verdict, Hebrew positive, English positive)``. Both positives, the
same text with the negative word removed or turned positive, accept.
"""

from __future__ import annotations

import bisect

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import hebrew_negators as negators
from axiom_encode.harness import hebrew_source_text as hebrew
from tests.test_source_completeness_hebrew import _accepted, _english_accepted

HEBREW_REASON = "חסר סעיף 2"
ENGLISH_REASON = "Missing section 2."


def _pair(hebrew_source: str, english_source: str) -> tuple[bool, bool]:
    return (
        _accepted("a", HEBREW_REASON, hebrew_source),
        _english_accepted("a", ENGLISH_REASON, english_source),
    )


IN_ACCORDANCE = " in accordance with section 2."

ROUND_22_REPRODUCERS = (
    # --- Finding 2: a relative only before an ``under`` linker -----------------
    (
        "f2-application-content",
        "(א) הטענה שהוגשה בקשה בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that an application was filed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהוגשה בקשה בהתאם לסעיף 2 נכונה.",
        "(a) It is true that an application was filed in accordance with section 2.",
    ),
    (
        "f2-discount-content",
        "(א) הטענה שניתנה הנחה בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that a discount was granted in accordance with section 2.",
        False,
        False,
        "(א) הטענה שניתנה הנחה בהתאם לסעיף 2 נכונה.",
        "(a) It is true that a discount was granted in accordance with section 2.",
    ),
    (
        "f2-reports-content",
        "(א) הטענות שהוגשו דוחות בהתאם לסעיף 2 אינן נכונות.",
        "(a) It is false that reports were filed in accordance with section 2.",
        False,
        False,
        "(א) הטענות שהוגשו דוחות בהתאם לסעיף 2 נכונות.",
        "(a) It is true that reports were filed in accordance with section 2.",
    ),
    (
        "f2-four-word-ki",
        "(א) הטענה שהועלתה אתמול בדיון רשמי נוסף כי התשלום מחושב בהתאם לסעיף 2 "
        "אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בדיון רשמי נוסף כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "f2-application-under",
        "(א) הטענה שהוגשה בקשה לפי סעיף 2 אינה נכונה.",
        "(a) It is false that an application was filed under section 2.",
        True,
        True,
        "(א) הטענה שהוגשה בקשה לפי סעיף 2 נכונה.",
        "(a) It is true that an application was filed under section 2.",
    ),
    (
        "f2-relative-in-accordance",
        "(א) הדיווחים שהוגשו בהתאם לסעיף 2 אינם נכונים.",
        "(a) The reports filed in accordance with section 2 are incorrect.",
        False,
        True,
        "(א) הדיווחים שהוגשו בהתאם לסעיף 2 נכונים.",
        "(a) The reports filed in accordance with section 2 are correct.",
    ),
    (
        "f2-content-under",
        "(א) הטענה שהתשלום מחושב לפי סעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed under section 2.",
        False,
        True,
        "(א) הטענה שהתשלום מחושב לפי סעיף 2 נכונה.",
        "(a) It is true that the payment is computed under section 2.",
    ),
    (
        "r10-no-denial",
        "(א) הטענה שהוגשה בקשה בהתאם לסעיף 2.",
        "(a) The claim that an application was filed in accordance with section 2.",
        True,
        True,
        "(א) הטענה שהוגשה בקשה בהתאם לסעיף 2.",
        "(a) The claim that an application was filed in accordance with section 2.",
    ),
    (
        "r10-agreement-control",
        "(א) הטענה שהוגש דוח בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that a report was filed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהוגש דוח בהתאם לסעיף 2 נכונה.",
        "(a) It is true that a report was filed in accordance with section 2.",
    ),
    # --- Finding 3: object-suffixed computation infinitives -------------------
    (
        "f3-calculate-it",
        "(א) הסכום ייקבע במקום לחשבו בהתאם לסעיף 2.",
        "(a) The amount is determined rather than calculating it" + IN_ACCORDANCE,
        False,
        False,
        "(א) הסכום ייקבע וכן לחשבו בהתאם לסעיף 2.",
        "(a) The amount is determined and calculated" + IN_ACCORDANCE,
    ),
    (
        "f3-apply-it",
        "(א) התעריף ייקבע במקום להחילו בהתאם לסעיף 2.",
        "(a) The tariff is determined rather than being applied" + IN_ACCORDANCE,
        False,
        False,
        "(א) התעריף ייקבע וכן להחילו בהתאם לסעיף 2.",
        "(a) The tariff is determined and applied" + IN_ACCORDANCE,
    ),
    (
        "f3-implement-it",
        "(א) ההסדר ייקבע במקום ליישמו בהתאם לסעיף 2.",
        "(a) The arrangement is determined rather than being applied" + IN_ACCORDANCE,
        False,
        False,
        "(א) ההסדר ייקבע וכן ליישמו בהתאם לסעיף 2.",
        "(a) The arrangement is determined and applied" + IN_ACCORDANCE,
    ),
    (
        "f3-impose-it",
        "(א) המס ייקבע במקום להטילו בהתאם לסעיף 2.",
        "(a) The tax is determined rather than being imposed" + IN_ACCORDANCE,
        False,
        False,
        "(א) המס ייקבע וכן להטילו בהתאם לסעיף 2.",
        "(a) The tax is determined and imposed" + IN_ACCORDANCE,
    ),
    (
        "f3-charge-him",
        "(א) הסכום ייקבע במקום לחייבו בהתאם לסעיף 2.",
        "(a) The amount is determined rather than being imposed on him" + IN_ACCORDANCE,
        False,
        False,
        "(א) הסכום ייקבע כדי לחייבו בהתאם לסעיף 2.",
        "(a) The amount is determined in order to impose it on him" + IN_ACCORDANCE,
    ),
    (
        "f3-charge-him-and",
        "(א) הסכום ייקבע במקום לחייבו בהתאם לסעיף 2.",
        "(a) The amount is determined rather than being imposed on him" + IN_ACCORDANCE,
        False,
        False,
        "(א) הסכום ייקבע וכן לחייבו בהתאם לסעיף 2.",
        "(a) The amount is determined and imposed on him" + IN_ACCORDANCE,
    ),
    (
        "f3-levy-it",
        "(א) המס ייקבע במקום להשיתו בהתאם לסעיף 2.",
        "(a) The tax is determined rather than being imposed" + IN_ACCORDANCE,
        False,
        False,
        "(א) המס ייקבע וכן להשיתו בהתאם לסעיף 2.",
        "(a) The tax is determined and imposed" + IN_ACCORDANCE,
    ),
    (
        "f3-charge-them",
        "(א) הסכומים ייקבעו במקום לחייבם בהתאם לסעיף 2.",
        "(a) The amounts are determined rather than being imposed on them"
        + IN_ACCORDANCE,
        False,
        False,
        "(א) הסכומים ייקבעו וכן לחייבם בהתאם לסעיף 2.",
        "(a) The amounts are determined and imposed on them" + IN_ACCORDANCE,
    ),
    (
        "r10-bare-infinitive-control",
        "(א) הסכום ייקבע במקום לחייב בהתאם לסעיף 2.",
        "(a) The amount is determined rather than being imposed" + IN_ACCORDANCE,
        False,
        False,
        "(א) הסכום ייקבע וכן לחייב בהתאם לסעיף 2.",
        "(a) The amount is determined and imposed" + IN_ACCORDANCE,
    ),
    # --- Finding 4: ``דוחות`` is "reports" without a feminine plural subject ---
    (
        "f4-filed-reports",
        "(א) יוגשו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        "(א) יוגשו מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents shall be filed in accordance with section 2.",
    ),
    (
        "f4-filed-definite-reports",
        "(א) יוגשו דוחות החישוב בהתאם לסעיף 2.",
        "(a) The calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        "(א) יוגשו מסמכים החישוב בהתאם לסעיף 2.",
        "(a) The calculation documents shall be filed in accordance with section 2.",
    ),
    (
        "f4-reports-first",
        "(א) דוחות חישוב יוגשו בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        "(א) יוגשו מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents shall be filed in accordance with section 2.",
    ),
    (
        "f4-taxpayer-files-reports",
        "(א) הנישום יגיש דוחות חישוב בהתאם לסעיף 2.",
        "(a) The taxpayer shall file calculation reports in accordance with section 2.",
        False,
        True,
        "(א) הנישום יגיש מסמכים חישוב בהתאם לסעיף 2.",
        "(a) The taxpayer shall file calculation documents in accordance with section 2.",
    ),
    (
        "f4-authorities-reject",
        "(א) הרשויות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות מבצעות חישוב בהתאם לסעיף 2.",
        "(a) The authorities perform calculation in accordance with section 2.",
    ),
    (
        "f4-they-reject",
        "(א) הן דוחות חישוב בהתאם לסעיף 2.",
        "(a) They reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הן מבצעות חישוב בהתאם לסעיף 2.",
        "(a) They perform calculation in accordance with section 2.",
    ),
    (
        "f4-authority-rejects",
        "(א) הרשות דוחה חישוב בהתאם לסעיף 2.",
        "(a) The authority rejects calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשות מבצעת חישוב בהתאם לסעיף 2.",
        "(a) The authority performs calculation in accordance with section 2.",
    ),
    (
        "f4-officials-reject",
        "(א) הפקידים דוחים חישוב בהתאם לסעיף 2.",
        "(a) The officials reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הפקידים מבצעים חישוב בהתאם לסעיף 2.",
        "(a) The officials perform calculation in accordance with section 2.",
    ),
)


@pytest.mark.parametrize(
    (
        "hebrew_source",
        "english_source",
        "hebrew_verdict",
        "english_verdict",
        "hebrew_positive",
        "english_positive",
    ),
    [row[1:] for row in ROUND_22_REPRODUCERS],
    ids=[row[0] for row in ROUND_22_REPRODUCERS],
)
def test_round_22_reproducers_and_their_matched_positive_controls(
    hebrew_source,
    english_source,
    hebrew_verdict,
    english_verdict,
    hebrew_positive,
    english_positive,
):
    assert _pair(hebrew_source, english_source) == (hebrew_verdict, english_verdict)
    assert _pair(hebrew_positive, english_positive) == (True, True)


def test_every_round_22_divergence_is_hebrew_stricter():
    """Invariant 1: no row accepts in Hebrew what its English refuses."""

    for row in ROUND_22_REPRODUCERS:
        assert row[3] <= row[4], row[0]
    assert {row[0] for row in ROUND_22_REPRODUCERS if row[3] != row[4]} == {
        "f2-relative-in-accordance",
        "f2-content-under",
        # Round 24 removes the clause-head noun context; guarded snapshot 0.
        "f4-reports-first",
        # Round 25: finite contexts removed; guarded snapshot count 0 each.
        "f4-filed-definite-reports",
        "f4-filed-reports",
        "f4-taxpayer-files-reports",
    }


# Intended stricter: before ``בהתאם ל``, a linker other than ``under``, a claim
# noun's ``ש`` opens a content clause, so a genuine relative refuses while its
# English accepts. The last two were round 21 controls that accepted.
INTENDED_STRICTER_RELATIVES = (
    (
        "(א) הדיווחים שהוגשו בהתאם לסעיף 2 אינם נכונים.",
        "(a) The reports filed in accordance with section 2 are incorrect.",
    ),
    (
        "(א) הטענה שהוגשה בהתאם לסעיף 2 אינה נכונה.",
        "(a) The claim filed in accordance with section 2 is incorrect.",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source"), INTENDED_STRICTER_RELATIVES
)
def test_relatives_before_in_accordance_are_intended_stricter(
    hebrew_source, english_source
):
    assert _pair(hebrew_source, english_source) == (False, True)


def test_a_relative_needs_an_under_linker_within_four_tokens():
    claim = hebrew._CLAIM_SUBJECT
    # Up to four tokens between the attributive verb and ``לפי``, then the
    # citation and the denial (round 23: the relative ends at its citation):
    # a relative.
    for run in ("", "בקשה", "בקשה אתמול", "בקשה אתמול בדיון רשמי"):
        text = f"הטענה שהוגשה {run} לפי סעיף 2 אינה נכונה".replace("  ", " ")
        assert claim.search(text) is None, text
    # Five tokens, another linker, no linker, or no denial after the
    # citation: content.
    for text in (
        "הטענה שהוגשה בקשה אתמול בדיון רשמי נוסף לפי סעיף 2 אינה נכונה",
        "הטענה שהוגשה בקשה בהתאם לסעיף 2 אינה נכונה",
        "הטענה שהוגשה בהתאם לסעיף 2 אינה נכונה",
        "הטענה שהוגשה אתמול",
        "הטענה שהוגשה לפי סעיף 2",
    ):
        assert claim.search(text) is not None, text
    # ``על פי`` and ``עפ"י`` are ``under`` linkers too.
    for linker in ("על פי", 'עפ"י', "ולפי"):
        text = f"הטענה שהוגשה בקשה {linker} סעיף 2 אינה נכונה"
        assert claim.search(text) is None, linker


# --- Finding 3: the generator's object suffixes ---------------------------------

TRANSITIVE = ("לחשב", "להחיל", "ליישם", "להטיל", "לחייב", "להשית")
INTRANSITIVE = ("להיקבע", "להיווצר", "לצמוח", "לחול", "לקום")


def test_transitive_infinitives_take_every_object_suffix():
    assert negators.OBJECT_SUFFIXES == ("ני", "ך", "ו", "ה", "נו", "כם", "כן", "ם", "ן")
    infinitives = set(negators.COMPUTATION_INFINITIVES)
    for infinitive in TRANSITIVE:
        assert infinitive in infinitives
        stem = infinitive.replace("ם", "מ")
        for suffix in negators.OBJECT_SUFFIXES:
            assert stem + suffix in infinitives, stem + suffix
    assert {"ליישמו", "ליישמם", "לחייבו", "להשיתן"} <= infinitives
    assert not {"ליישםו", "ליישםם"} & infinitives
    for infinitive in INTRANSITIVE:
        assert infinitive in infinitives
        assert (
            not {infinitive + suffix for suffix in negators.OBJECT_SUFFIXES}
            & infinitives
        )
    assert len(infinitives) == len(TRANSITIVE) * 10 + len(INTRANSITIVE)


def test_the_snapshot_charge_him_keeps_its_context_guard():
    """National Insurance §369: ``הכוונה לחייבו``, "the intent to charge him"."""

    source = (
        "(א) רשאי הוא לטעון את טענותיו לעניין הכוונה לחייבו ולעניין סכומו "
        "בהתאם לסעיף 2."
    )
    assert _accepted("a", HEBREW_REASON, source)


# --- Finding 4: ``דוחות`` is the verb unless a noun context applies -----------
# Round 23 (review r11 B2): the verb is the default, so a bare clause-initial
# ``דוחות חישוב`` and any subject read "reject"; only a closed noun context
# (a listed preposition or filing construct) reads "reports". Round 25
# removes finite filing contexts; those rows intentionally refuse.


@pytest.mark.parametrize(
    ("subject", "negates"),
    (
        ("הרשויות", True),
        ("והרשויות", True),
        ("הן", True),
        ("אלה", True),
        ("אלו", True),
        ("הנשים", True),
        ("והן", True),
        ("ואלה", True),
        ("יוגשו", True),
        ("הנישום יגיש", True),
        ("", True),
    ),
)
def test_reject_is_the_verb_unless_a_noun_context_applies(subject, negates):
    source = f"(א) {subject} דוחות חישוב בהתאם לסעיף 2.".replace("  ", " ")
    assert (not hebrew._linked_source_instruments(source)) is negates, source


def test_the_filter_is_the_source_readers():
    """A reason's ``דוחות`` reads as before; ``דוחות את`` stays the verb."""

    assert hebrew._REASON_NEGATION._source is False
    assert hebrew._NEGATIONS._source is True
    assert hebrew._REASON_NEGATION.search("(א) דוחות את הטענה") is not None
    assert hebrew._NEGATIONS.search("(א) דוחות את הטענה") is not None


# --- Finding 5: forward cursors in the citation loop ----------------------------


@settings(max_examples=500, deadline=None)
@given(
    items=st.lists(st.integers(min_value=-40, max_value=40), max_size=40).map(sorted),
    queries=st.lists(st.integers(min_value=-50, max_value=50), max_size=60),
    forward=st.booleans(),
    right=st.booleans(),
)
def test_the_bisect_cursor_equals_bisect_in_any_query_order(
    items, queries, forward, right
):
    """Differential: any sorted array, any query order, forward or not."""

    if forward:
        queries = sorted(queries)
    cursor = hebrew._BisectCursor(items, right=right)
    search = bisect.bisect_right if right else bisect.bisect_left
    for query in queries:
        assert cursor.index(query) == search(items, query), (items, query)


class _Counted:
    """A sequence that counts its indexed reads."""

    def __init__(self, items, reads):
        self._items = items
        self._reads = reads

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        self._reads[0] += 1
        return self._items[index]


def _reads(n: int, monkeypatch) -> int:
    """Indexed reads of every array the reader searches, as review r10 counted."""

    reads = [0]

    def counting(original):
        def call(items, query, lo=0, hi=None):
            if not isinstance(items, _Counted):
                items = _Counted(items, reads)
            return original(items, query, lo, len(items) if hi is None else hi)

        return call

    class CountingCursor(hebrew._BisectCursor):
        __slots__ = ()

        def __init__(self, items, *, right=False):
            super().__init__(_Counted(items, reads), right=right)

    with monkeypatch.context() as patch:
        patch.setattr(bisect, "bisect_left", counting(bisect.bisect_left))
        patch.setattr(bisect, "bisect_right", counting(bisect.bisect_right))
        patch.setattr(hebrew, "_BisectCursor", CountingCursor)
        index = hebrew._linked_source_instruments(
            "נטול חישוב בהתאם לסעיף 2 " * n + "סוף."
        )
    assert index == {}
    return reads[0]


# Review r10's counts at the growth base 8cb27cfd7, whose reader found no
# negator in this family: 2n reads (growth-counts-base.json).
BASE_READS = {8000: 16000, 16000: 32000, 32000: 64000}


def test_the_privative_family_reads_linearly(monkeypatch):
    """``נטול חישוב בהתאם לסעיף 2`` × n: reads within a constant of the base's."""

    counts = {n: _reads(n, monkeypatch) for n in BASE_READS}
    for n, base in BASE_READS.items():
        assert counts[n] <= 3 * base, (n, counts[n])
    assert counts[16000] <= 2 * counts[8000] + 16
    assert counts[32000] <= 2 * counts[16000] + 16


# --- Left open ---------------------------------------------------------------------
#
# Review r10's non-blocking rows the PR leaves open. Each holds invariant 1's
# target and is expected to fail until the open item closes.
LEFT_OPEN = (
    (
        "report-claim-noun",
        "(א) הדוח שהתשלום מחושב בהתאם לסעיף 2 אינו נכון.",
        "(a) It is false that the payment is computed in accordance with section 2.",
    ),
    (
        "saying-homograph",
        "(א) האמרה שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
    ),
)


@pytest.mark.xfail(
    strict=True,
    reason="left open in #1951: `דוח` and `אמרה` sit outside the claim families",
)
@pytest.mark.parametrize(
    ("hebrew_source", "english_source"),
    [row[1:] for row in LEFT_OPEN],
    ids=[row[0] for row in LEFT_OPEN],
)
def test_left_open_claim_nouns(hebrew_source, english_source):
    assert _pair(hebrew_source, english_source) == (False, False)
