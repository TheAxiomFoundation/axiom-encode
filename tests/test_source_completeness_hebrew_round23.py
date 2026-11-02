"""Round 23 of #1882: one gate, the verb by default, relatives that end.

Review r11 (GPT-6.1 Sol, head 1f297578e) found the wall broken through a
rule's proof excerpt (B1; the gate itself is checked in
tests/test_source_completeness_hebrew_gate.py and every other rule field in
the wall module's channel fixture), false accepts from reading ``דוחות`` as
the noun "reports" by default (B2), and a relative with its own ``under``
citation followed by the claim's content (B3). Every expectation here is
written by hand from the targets the main session set, not derived from the
generator.

A row is ``(id, Hebrew source, faithful English source, Hebrew verdict,
English verdict, Hebrew positive, English positive)``. Both positives, the
same text with the negative word turned positive or the denial affirmed,
accept.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness import hebrew_negators as negators
from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_source_completeness_hebrew import _accepted, _english_accepted

HEBREW_REASON = "חסר סעיף 2"
ENGLISH_REASON = "Missing section 2."


def _pair(hebrew_source: str, english_source: str) -> tuple[bool, bool]:
    return (
        _accepted("a", HEBREW_REASON, hebrew_source),
        _english_accepted("a", ENGLISH_REASON, english_source),
    )


# B2 and N4: the guarded ``דוחות`` and the verb-first ``דוחות … את`` are the
# verb; the noun contexts accept. B3: the relative's own ``under`` citation,
# then anything but the denial, opens content. ``r11-*`` rows are every row
# of review r11's linguistic sweeps (r11/ling.py, ling2.py) that round 22
# accepted while English refused, but the two left open below.
ROUND_23_REPRODUCERS = (
    (
        "b2-women",
        "(א) הנשים דוחות חישוב בהתאם לסעיף 2.",
        "(a) The women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "b2-and-they",
        "(א) והן דוחות חישוב בהתאם לסעיף 2.",
        "(a) And they reject calculation in accordance with section 2.",
        False,
        False,
        "(א) והן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) And they accept calculation in accordance with section 2.",
    ),
    (
        "b2-and-these",
        "(א) ואלה דוחות חישוב בהתאם לסעיף 2.",
        "(a) And these reject calculation in accordance with section 2.",
        False,
        False,
        "(א) ואלה מקבלות חישוב בהתאם לסעיף 2.",
        "(a) And these accept calculation in accordance with section 2.",
    ),
    (
        "b2-maqaf",
        "(א) הרשויות־דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות־מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities accept calculation in accordance with section 2.",
    ),
    (
        "b2-all",
        "(א) הרשויות כולן דוחות חישוב בהתאם לסעיף 2.",
        "(a) All the authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות כולן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) All the authorities accept calculation in accordance with section 2.",
    ),
    (
        "b2-these",
        "(א) הרשויות האלה דוחות חישוב בהתאם לסעיף 2.",
        "(a) These authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות האלה מקבלות חישוב בהתאם לסעיף 2.",
        "(a) These authorities accept calculation in accordance with section 2.",
    ),
    (
        "b2-listed-subject",
        "(א) הרשויות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities accept calculation in accordance with section 2.",
    ),
    (
        "b2-listed-pronoun",
        "(א) הן דוחות חישוב בהתאם לסעיף 2.",
        "(a) They reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) They accept calculation in accordance with section 2.",
    ),
    (
        "b2-maqaf-space",
        "(א) הרשויות־ דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות־ מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities accept calculation in accordance with section 2.",
    ),
    (
        "b2-verb-first",
        "(א) דוחות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities accept the calculation in accordance with section 2.",
    ),
    (
        "b2-noun-filed",
        "(א) יוגשו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        True,
        True,
        "(א) יוגשו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
    ),
    (
        "b2-noun-filed-definite",
        "(א) יוגשו דוחות החישוב בהתאם לסעיף 2.",
        "(a) The calculation reports shall be filed in accordance with section 2.",
        True,
        True,
        "(א) יוגשו דוחות החישוב בהתאם לסעיף 2.",
        "(a) The calculation reports shall be filed in accordance with section 2.",
    ),
    (
        "b2-noun-clause-head",
        "(א) דוחות חישוב יוגשו בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        True,
        True,
        "(א) דוחות חישוב יוגשו בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
    ),
    (
        "b2-noun-taxpayer-files",
        "(א) הנישום יגיש דוחות חישוב בהתאם לסעיף 2.",
        "(a) The taxpayer shall file calculation reports in accordance with section 2.",
        True,
        True,
        "(א) הנישום יגיש דוחות חישוב בהתאם לסעיף 2.",
        "(a) The taxpayer shall file calculation reports in accordance with section 2.",
    ),
    (
        "b2-noun-authorities-file",
        "(א) הרשויות מגישות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities file calculation reports in accordance with section 2.",
        True,
        True,
        "(א) הרשויות מגישות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities file calculation reports in accordance with section 2.",
    ),
    (
        "b2-noun-funds-prepare",
        "(א) הקופות מכינות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The funds prepare calculation reports in accordance with section 2.",
        True,
        True,
        "(א) הקופות מכינות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The funds prepare calculation reports in accordance with section 2.",
    ),
    (
        "r10-rejects-singular",
        "(א) הרשות דוחה חישוב בהתאם לסעיף 2.",
        "(a) The authority rejects calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשות מקבלת חישוב בהתאם לסעיף 2.",
        "(a) The authority accepts calculation in accordance with section 2.",
    ),
    (
        "r10-reject-masculine",
        "(א) הפקידים דוחים חישוב בהתאם לסעיף 2.",
        "(a) The officials reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הפקידים מקבלים חישוב בהתאם לסעיף 2.",
        "(a) The officials accept calculation in accordance with section 2.",
    ),
    (
        "b3-run0-digit-ki",
        "(א) הטענה שהועלתה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-digit-and-ki",
        "(א) הטענה שהועלתה לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-digit-she",
        "(א) הטענה שהועלתה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-letter-ki",
        "(א) הטענה שהועלתה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-letter-and-ki",
        "(א) הטענה שהועלתה לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-letter-she",
        "(א) הטענה שהועלתה לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-law-ki",
        "(א) הטענה שהועלתה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-law-and-ki",
        "(א) הטענה שהועלתה לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-law-she",
        "(א) הטענה שהועלתה לפי החוק שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי החוק שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-fragment-ki",
        "(א) הטענה שהועלתה לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-fragment-and-ki",
        "(א) הטענה שהועלתה לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-fragment-she",
        "(א) הטענה שהועלתה לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-two-ki",
        "(א) הטענה שהועלתה לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-two-and-ki",
        "(א) הטענה שהועלתה לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run0-two-she",
        "(א) הטענה שהועלתה לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-digit-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-digit-and-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-digit-she",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-letter-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-letter-and-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-letter-she",
        "(א) הטענה שהועלתה אתמול לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-law-ki",
        "(א) הטענה שהועלתה אתמול לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-law-and-ki",
        "(א) הטענה שהועלתה אתמול לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-law-she",
        "(א) הטענה שהועלתה אתמול לפי החוק שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי החוק שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-fragment-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-fragment-and-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-fragment-she",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-two-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-two-and-ki",
        "(א) הטענה שהועלתה אתמול לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run1-two-she",
        "(א) הטענה שהועלתה אתמול לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-digit-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-digit-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-digit-she",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-letter-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-letter-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-letter-she",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-law-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-law-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-law-she",
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-fragment-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-fragment-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-fragment-she",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-two-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-two-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run2-two-she",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-digit-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-digit-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-digit-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-letter-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-letter-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-letter-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-law-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-law-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-law-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-fragment-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-fragment-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-fragment-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-two-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-two-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run3-two-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-digit-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-digit-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-digit-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-letter-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-letter-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-letter-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-law-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-law-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-law-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-fragment-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-fragment-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-fragment-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-two-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-two-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run4-two-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-digit-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-digit-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-digit-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-letter-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-letter-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף ג וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-letter-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף ג שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-law-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-law-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי החוק וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-law-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי החוק שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי החוק שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-fragment-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3(א) כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-fragment-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3(א) וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-fragment-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 3(א) שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-two-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיפים 3 ו-4 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-two-and-ki",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיפים 3 ו-4 וכי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-run5-two-she",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיפים 3 ו-4 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-plural",
        "(א) הטענות שהועלו לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינן נכונות.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענות שהועלו לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונות.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-maqaf-in-run",
        "(א) הטענה שהועלתה לפי־סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי־סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-maqaf-after-verb",
        "(א) הטענה שהוגשה־אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהוגשה־אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-affirmed-control",
        "(א) הטענה שהועלתה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
        True,
        True,
        "(א) הטענה שהועלתה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "b3-relative-only",
        "(א) הטענה שהוגשה לפי סעיף 2 אינה נכונה.",
        "(a) The claim filed under section 2 is false.",
        True,
        True,
        "(א) הטענה שהוגשה לפי סעיף 2 נכונה.",
        "(a) The claim filed under section 2 is true.",
    ),
    (
        "b3-relative-only-plural",
        "(א) הדיווחים שהוגשו לפי סעיף 2 אינם נכונים.",
        "(a) The reports filed under section 2 are incorrect.",
        True,
        True,
        "(א) הדיווחים שהוגשו לפי סעיף 2 נכונים.",
        "(a) The reports filed under section 2 are correct.",
    ),
    (
        "f2-four-word-ki",
        "(א) הטענה שהועלתה אתמול בדיון רשמי נוסף כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בדיון רשמי נוסף כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that the payment is computed in accordance with section 2.",
    ),
    (
        "r11-fix4-subject",
        "(א) הנשים דוחות חישוב בהתאם לסעיף 2.",
        "(a) The women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-1",
        "(א) הערים דוחות חישוב בהתאם לסעיף 2.",
        "(a) The cities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הערים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The cities accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-2",
        "(א) השנים דוחות חישוב בהתאם לסעיף 2.",
        "(a) The years reject calculation in accordance with section 2.",
        False,
        False,
        "(א) השנים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The years accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-3",
        "(א) והן דוחות חישוב בהתאם לסעיף 2.",
        "(a) And they reject calculation in accordance with section 2.",
        False,
        False,
        "(א) והן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) And they accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-4",
        "(א) שהן דוחות חישוב בהתאם לסעיף 2.",
        "(a) That they reject calculation in accordance with section 2.",
        False,
        False,
        "(א) שהן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) That they accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-5",
        "(א) כשהן דוחות חישוב בהתאם לסעיף 2.",
        "(a) When they reject calculation in accordance with section 2.",
        False,
        False,
        "(א) כשהן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) When they accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-6",
        "(א) ואלה דוחות חישוב בהתאם לסעיף 2.",
        "(a) And these reject calculation in accordance with section 2.",
        False,
        False,
        "(א) ואלה מקבלות חישוב בהתאם לסעיף 2.",
        "(a) And these accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-7",
        "(א) ואלו דוחות חישוב בהתאם לסעיף 2.",
        "(a) And these reject calculation in accordance with section 2.",
        False,
        False,
        "(א) ואלו מקבלות חישוב בהתאם לסעיף 2.",
        "(a) And these accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-8",
        "(א) הללו דוחות חישוב בהתאם לסעיף 2.",
        "(a) These reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הללו מקבלות חישוב בהתאם לסעיף 2.",
        "(a) These accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-9",
        "(א) הרשויות האלה דוחות חישוב בהתאם לסעיף 2.",
        "(a) These authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות האלה מקבלות חישוב בהתאם לסעיף 2.",
        "(a) These authorities accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-10",
        "(א) הרשויות הללו דוחות חישוב בהתאם לסעיף 2.",
        "(a) These authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות הללו מקבלות חישוב בהתאם לסעיף 2.",
        "(a) These authorities accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-11",
        "(א) הרשויות בישראל דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities in Israel reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות בישראל מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities in Israel accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-12",
        "(א) הרשויות כולן דוחות חישוב בהתאם לסעיף 2.",
        "(a) All the authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות כולן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) All the authorities accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-13",
        "(א) הרשויות עצמן דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities themselves reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות עצמן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities themselves accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-14",
        "(א) הרשויות תמיד דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities always reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות תמיד מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities always accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-15",
        "(א) הרשויות גם דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities also reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות גם מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities also accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-16",
        "(א) הרשויות והנשים דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities and the women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות והנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities and the women accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-17",
        "(א) הרשויות, דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות, מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-18",
        "(א) הרשויות - דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות - מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-subject-19",
        "(א) הרשויות־ דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות־ מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-verbform",
        "(א) הנשים דּוֹחוֹת חישוב בהתאם לסעיף 2.",
        "(a) The women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-verbform-1",
        "(א) הנשים דוחוֹת חישוב בהתאם לסעיף 2.",
        "(a) The women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-verbform-2",
        "(א) הנשים ודוחות חישוב בהתאם לסעיף 2.",
        "(a) The women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-verbform-3",
        "(א) הנשים שדוחות חישוב בהתאם לסעיף 2.",
        "(a) The women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-verbform-4",
        "(א) הנשים דוחות‍ חישוב בהתאם לסעיף 2.",
        "(a) The women reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-order",
        "(א) דוחות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities accept the calculation in accordance with section 2.",
    ),
    (
        "r11-fix4-order-1",
        "(א) דוחות הן את החישוב בהתאם לסעיף 2.",
        "(a) They reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות הן את החישוב בהתאם לסעיף 2.",
        "(a) They accept the calculation in accordance with section 2.",
    ),
    (
        "r11-rel-window-0",
        "(א) הטענה שהועלתה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-0-1",
        "(א) הטענה שהועלתה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-0-2",
        "(א) הטענה שהועלתה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-0-3",
        "(א) הטענה שהועלתה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised under the law, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised under the law, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-1",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-1-1",
        "(א) הטענה שהועלתה אתמול לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-1-2",
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-2",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-2-1",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-2-2",
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-2-3",
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning under the law, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning under the law, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-3",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the municipality under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the municipality under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-3-1",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the municipality under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the municipality under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-3-2",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the municipality under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the municipality under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-3-3",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the municipality under the law, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the municipality under the law, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-4",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the central municipality under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the central municipality under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-4-1",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the central municipality under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף ג כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the central municipality under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-4-2",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the central municipality under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the central municipality under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-window-4-3",
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised yesterday morning at the central municipality under the law, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה אתמול בבוקר בעירייה המרכזית לפי החוק כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised yesterday morning at the central municipality under the law, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-inside",
        "(א) הטענה שהוגשה־אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim filed yesterday under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהוגשה־אתמול לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim filed yesterday under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-maqaf-nospace",
        "(א) הרשויות־דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הרשויות־מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The authorities accept calculation in accordance with section 2.",
    ),
    (
        "r11-women-canon-false",
        "(א) הנשים דוחות חישוב בהתאם לסעיף 2.",
        "(a) It is false that the women accept calculation in accordance with section 2.",
        False,
        False,
        "(א) הנשים מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The women accept calculation in accordance with section 2.",
    ),
    (
        "r11-and-they-statute",
        "(א) הקופות יבחנו את הבקשה, והן דוחות חישוב בהתאם לסעיף 2.",
        "(a) The funds shall examine the application, and they reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הקופות יבחנו את הבקשה, והן מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The funds shall examine the application, and they accept calculation in accordance with section 2.",
    ),
    (
        "r11-rel-masc-pl",
        "(א) הטענות שהועלו לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינן נכונות.",
        "(a) It is false that, as asserted in the claims raised under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענות שהועלו לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכונות.",
        "(a) It is true that, as asserted in the claims raised under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-masc-sg",
        "(א) הדיווח שהוגש לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינו נכון.",
        "(a) It is false that, as asserted in the report filed under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הדיווח שהוגש לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 נכון.",
        "(a) It is true that, as asserted in the report filed under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-ki-direct",
        "(א) הטענה שהועלתה לפי סעיף 3, כי התשלום מחושב בהתאם לסעיף 2, אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3, כי התשלום מחושב בהתאם לסעיף 2, נכונה.",
        "(a) It is true that, as asserted in the claim raised under section 3, the payment is computed in accordance with section 2.",
    ),
    (
        "r11-rel-ki-adverb",
        "(א) הטענה שהועלתה לפי סעיף 3 בטעות כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה.",
        "(a) It is false that, as asserted in the claim raised under section 3 by mistake, the payment is computed in accordance with section 2.",
        False,
        False,
        "(א) הטענה שהועלתה לפי סעיף 3 בטעות כי התשלום מחושב בהתאם לסעיף 2 נכונה.",
        "(a) It is true that, as asserted in the claim raised under section 3 by mistake, the payment is computed in accordance with section 2.",
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
    [row[1:] for row in ROUND_23_REPRODUCERS],
    ids=[row[0] for row in ROUND_23_REPRODUCERS],
)
def test_round_23_reproducers_and_their_matched_positive_controls(
    hebrew_source,
    english_source,
    hebrew_verdict,
    english_verdict,
    hebrew_positive,
    english_positive,
):
    assert _pair(hebrew_source, english_source) == (hebrew_verdict, english_verdict)
    assert _pair(hebrew_positive, english_positive) == (True, True)


def test_every_round_23_reproducer_is_no_looser_than_english():
    for row in ROUND_23_REPRODUCERS:
        hebrew_verdict, english_verdict = row[3], row[4]
        assert not (hebrew_verdict and not english_verdict), row[0]


# Hebrew refuses where canonical English accepts, on purpose: a false refusal
# with a label does not block and a false accept does. The count is the
# construction's occurrences in the pinned 1,633-record snapshot.
INTENDED_STRICTER_ROUND_23 = (
    (
        "n1-notices-and-reports",
        "(א) יוגשו הודעות ודוחות חישוב בהתאם לסעיף 2.",
        "(a) Notices and calculation reports shall be filed in accordance with section 2.",
        0,
    ),
    (
        "n1-reviews-of-reports",
        "(א) יבוצעו בדיקות דוחות חישוב בהתאם לסעיף 2.",
        "(a) Reviews of calculation reports shall be carried out in accordance with section 2.",
        0,
    ),
    (
        "n1-copies-of-reports",
        "(א) יוגשו העתקות דוחות חישוב בהתאם לסעיף 2.",
        "(a) Copies of calculation reports shall be filed in accordance with section 2.",
        0,
    ),
    (
        "n1-provisions-of-reports",
        "(א) לפי הוראות דוחות חישוב בהתאם לסעיף 2.",
        "(a) Under the provisions of calculation reports in accordance with section 2.",
        0,
    ),
    (
        "n2-five-tokens-before-under",
        "(א) הטענה שהוגשה אתמול בבוקר בעירייה המרכזית בעיר לפי סעיף 2 אינה נכונה.",
        "(a) The claim filed yesterday morning at the central municipality in the "
        "city under section 2 is false.",
        0,
    ),
    (
        "relative-unbounded-citation",
        "(א) הטענה שהוגשה לפי החוק בהתאם לסעיף 2 אינה נכונה.",
        "(a) The claim filed under the law in accordance with section 2 is false.",
        0,
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source", "snapshot_count"),
    [row[1:] for row in INTENDED_STRICTER_ROUND_23],
    ids=[row[0] for row in INTENDED_STRICTER_ROUND_23],
)
def test_round_23_intended_stricter_refusals(
    hebrew_source, english_source, snapshot_count
):
    assert _pair(hebrew_source, english_source) == (False, True)
    assert snapshot_count == 0


# --- B2: the closed noun contexts ---------------------------------------------


def _links(source: str) -> bool:
    return bool(hebrew._linked_source_instruments(source))


@pytest.mark.parametrize(
    "before",
    (
        *sorted(negators.REPORTS_PREPOSITIONS),
        "ולענין",
        *sorted(negators.REPORTS_CONSTRUCTS),
        "ולהגשת",
        "יוגשו",
        "מגישות",
        "להכין",
        "נערכים",
        "מכינות",
        "פורסמו",
        "שקיבלו",
        "יתקבלו",
        "נמסרים",
        "תוכנה",
    ),
)
def test_a_noun_context_reads_reports(before):
    assert _links(f"(א) {before} דוחות חישוב בהתאם לסעיף 2.")


@pytest.mark.parametrize(
    "before",
    (
        "הרשויות",
        "הנשים",
        "והן",
        "הרשויות־",
        "הרשויות־ ",
        "הרשויות,",
        "הרשויות -",
        "הודעות ו",
        "בדיקות",
        "כי",
        "הנישום",
        "יבוצע",
    ),
)
def test_every_other_context_reads_reject(before):
    source = f"(א) {before} דוחות חישוב בהתאם לסעיף 2.".replace("־ ד", "־ד")
    if before == "הרשויות־ ":
        source = "(א) הרשויות־ דוחות חישוב בהתאם לסעיף 2."
    if before == "הודעות ו":
        source = "(א) הודעות ודוחות חישוב בהתאם לסעיף 2."
    assert not _links(source), source


def test_a_clause_head_reads_reports_only_before_a_filing_verb():
    assert _links("(א) דוחות חישוב יוגשו בהתאם לסעיף 2.")
    assert _links("(8) דוחות שיגישו החברה בהתאם לסעיף 2.")
    assert _links("דוחות חישוב יוגשו בהתאם לסעיף 2.")
    assert not _links("(א) דוחות חישוב בהתאם לסעיף 2.")
    assert not _links("(א) דוחות חישוב בהתאם לסעיף 2 יוגשו.")
    assert not _links("(א) דוחות חישוב יוגשה בהתאם לסעיף 2.")


def test_a_clause_head_reads_a_bounded_window():
    """A clause-initial ``דוחות`` reads a constant window, not the rest of the text."""

    reads = []

    class Recording(str):
        def __getitem__(self, key):
            if isinstance(key, slice):
                start, stop, _ = key.indices(len(self))
                reads.append(stop - start)
            return str.__getitem__(self, key)

    text = Recording("(1) דוחות חישוב יוגשו בהתאם לסעיף 2; " + "מילה " * 20000)
    start = text.index("דוחות")
    assert negators._reports_clause_subject(text, start, start + len("דוחות"))
    assert reads and max(reads) <= negators._REPORTS_SUBJECT_WINDOW


def test_the_filing_paradigms_are_generated():
    forms = {form for verb in negators.REPORTS_FILING_VERBS for form in verb.forms}
    for form in (
        "הגיש",
        "הגישה",
        "הגישו",
        "מגיש",
        "מגישה",
        "מגישים",
        "מגישות",
        "יגיש",
        "תגיש",
        "יגישו",
        "להגיש",
        "הוגש",
        "מוגשת",
        "יוגשו",
        "תוגשנה",
        "הכין",
        "מכינה",
        "יכינו",
        "להכין",
        "הוכנו",
        "מוכנה",
        "ערך",
        "עורכת",
        "יערכו",
        "תערכנה",
        "לערוך",
        "נערכים",
        "ייערכו",
        "מסר",
        "מוסרים",
        "ימסרו",
        "תמסרנה",
        "למסור",
        "נמסרה",
        "יימסרו",
        "קיבל",
        "מקבלות",
        "יקבלו",
        "לקבל",
        "התקבלו",
        "מתקבלים",
        "פרסם",
        "מפרסמת",
        "יפרסמו",
        "פורסמו",
        "מפורסמים",
        "יפורסמו",
    ):
        assert form in forms, form


def test_the_reason_reader_is_unchanged_by_the_verb_default():
    assert hebrew._REASON_NEGATION._source is False
    assert hebrew._REASON_NEGATION.search("(א) הנשים דוחות חישוב") is None
    assert hebrew._NEGATIONS.search("(א) הנשים דוחות חישוב") is not None


# --- B3: the relative's span -------------------------------------------------


def test_a_relative_reads_only_its_own_span():
    claim = hebrew._CLAIM_SUBJECT
    assert claim.search("הטענה שהוגשה לפי סעיף 2 אינה נכונה") is None
    assert claim.search("הטענה שהוגשה לפי סעיף 345ב(א) אינה נכונה") is None
    assert claim.search("ההנחות שיינתנו לפי סעיף 2 אינן נכונות") is None
    for text in (
        "הטענה שהוגשה לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינה נכונה",
        "הטענה שהוגשה לפי סעיף 3 וכי התשלום מחושב בהתאם לסעיף 2 אינה נכונה",
        "הטענה שהוגשה לפי סעיף 3 שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה",
        "הטענה שהוגשה לפי סעיף 3 אתמול אינה נכונה",
        "הטענה שהוגשה לפי סעיף ג אינה נכונה",
        "הטענה שהוגשה לפי החוק אינה נכונה",
        "הטענה שהוגשה לפי סעיפים 3 ו-4 אינה נכונה",
        "ההנחות שיינתנו לפי סעיף 3 כי התשלום מחושב בהתאם לסעיף 2 אינן נכונות",
    ):
        assert claim.search(text) is not None, text


def test_claim_content_needs_the_relatives_end():
    with pytest.raises(ValueError):
        negators.claim_content(after="", separator=r"\s+", exclude_relatives=True)


# --- B1: the excerpt reproducer's five variants, main's arrays ----------------

EXCERPT_SOURCE = "(1) Die Steuer ist das Einkommen multipliziert mit 10 Prozent."
EXCERPT_CANDIDATE = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/statute/estg/32a
rules:
  - name: tariff_rate
    kind: parameter
    dtype: Rate
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: rate
            source:
              corpus_citation_path: de/statute/estg/32a
              excerpt: "Die Steuer ist das Einkommen multipliziert mit 10 Prozent."
    versions:
      - effective_from: '2026-01-01'
        formula: 0.1
  - name: tax
    kind: derived
    dtype: Money
    source: de/statute/estg/32a(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: CITE
              excerpt: "EXCERPT"
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_income * tariff_rate
"""
EXCERPT_TESTS = [
    {
        "name": "t",
        "period": "2026",
        "input": {"taxable_income": 100},
        "output": {"de:statutes/estg/32a#tax": 10},
    }
]
EXCERPT_UNDEMONSTRATED = (
    "[complete-source-unit:tests] Companion tests do not demonstrate formula "
    "branch (1) formula clause 1 at de/statute/estg/32a(1) [Absatz 1]. Exact "
    "source computation (characters 0:62): `(1) Die Steuer ist das Einkommen "
    "multipliziert mit 10 Prozent.`. The asserted principal formula must execute "
    "this source computation in its legally applicable companion-case period. "
    "Observed asserted candidate-case periods: 2026-01-01. Each formula branch "
    "needs distinct executed test evidence; an internal formula-clause ordinal is "
    "not a legal paragraph number."
)
# Main's arrays (5d40209bc, unchanged at a86a37e71), from review r11's
# r11/excerpt_repro.py.
EXCERPT_VARIANTS = (
    (
        "control-plain-prefix-other-unit",
        "Die Steuer ist das Einkommen multipliziert",
        "de/statute/estg/32b",
        [EXCERPT_UNDEMONSTRATED],
    ),
    (
        "hebrew-label-prefix-other-unit",
        "(א) Die Steuer ist das Einkommen multipliziert",
        "de/statute/estg/32b",
        [],
    ),
    (
        "latin-label-prefix-other-unit",
        "(a) Die Steuer ist das Einkommen multipliziert",
        "de/statute/estg/32b",
        [EXCERPT_UNDEMONSTRATED],
    ),
    (
        "hebrew-label-prefix-same-unit",
        "(א) Die Steuer ist das Einkommen multipliziert",
        "de/statute/estg/32a",
        [],
    ),
    (
        "pointed-hebrew-label-prefix-other-unit",
        "(אֲ) Die Steuer ist das Einkommen multipliziert",
        "de/statute/estg/32b",
        [],
    ),
)


@pytest.mark.parametrize(
    ("excerpt", "citation", "main_issues"),
    [row[1:] for row in EXCERPT_VARIANTS],
    ids=[row[0] for row in EXCERPT_VARIANTS],
)
def test_a_hebrew_label_on_an_excerpt_takes_mains_array(excerpt, citation, main_issues):
    import functools

    content = EXCERPT_CANDIDATE.replace("EXCERPT", excerpt).replace("CITE", citation)
    issues = analyze_complete_source_unit(
        content,
        EXCERPT_SOURCE,
        corpus_citation_path="de/statute/estg/32a",
        test_cases=EXCERPT_TESTS,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="de-DE"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="de-DE"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues
    assert list(issues) == main_issues


# --- Left open ------------------------------------------------------------------
# ``תשלום`` is no computation head, so ``דוחות תשלום`` is not guarded, as the
# stated ``מונע תשלום`` boundary is not (accepting since 5dcf2bc13).


@pytest.mark.xfail(strict=True, reason="the stated computation-head boundary")
def test_reject_payment_stays_the_stated_boundary():
    assert _pair(
        "(א) הקופות כולן דוחות תשלום בהתאם לסעיף 2.",
        "(a) All the funds reject payment in accordance with section 2.",
    ) == (False, False)
