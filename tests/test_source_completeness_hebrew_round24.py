"""Round 24: remove unneeded noun contexts instead of parsing their syntax.

Expectations are hand-written from the review targets. Clause-initial reports
and reports after a present filing form intentionally refuse: neither shape
occurs before a guarded link in the 1,633-record snapshot (25 reports tokens).
Round 25 also removes finite filing contexts (intended-stricter, guarded
snapshot count 0 each). Every row retains an accepting document control.
"""

from __future__ import annotations

import pytest

from tests.hebrew_reports_snapshot import guarded_snapshot_count
from tests.test_source_completeness_hebrew import _accepted, _english_accepted

FILED_H = "(א) יוגשו מסמכים לחישוב בהתאם לסעיף 2."
FILED_E = "(a) Calculation documents shall be filed in accordance with section 2."
RELATIVE_E = (
    "(a) Calculation reports which shall be filed in accordance with section 2."
)

# (id, Hebrew, faithful English, H target, E target, H positive, E positive).
# The nine repro_blockers.py rows and both minimized scripts are included;
# repeated source rows retain their review names so provenance is explicit.
ROUND_24_REPRODUCERS = (
    (
        "b1-object-relative",
        "(א) דוחות הרשויות את החישובים שנמסרו בהתאם לסעיף 2.",
        "(a) The authorities reject the calculations that were submitted in accordance with section 2.",
        False,
        False,
        "(א) מקבלות הרשויות את החישובים שנמסרו בהתאם לסעיף 2.",
        "(a) The authorities accept the calculations that were submitted in accordance with section 2.",
    ),
    (
        "b1-indefinite-object-relative",
        "(א) דוחות הרשויות חישובים שנמסרו בהתאם לסעיף 2.",
        "(a) The authorities reject calculations that were submitted in accordance with section 2.",
        False,
        False,
        "(א) מקבלות הרשויות חישובים שנמסרו בהתאם לסעיף 2.",
        "(a) The authorities accept calculations that were submitted in accordance with section 2.",
    ),
    (
        "noun-control-intended-stricter",
        "(א) דוחות חישוב של הנישום יוגשו בהתאם לסעיף 2.",
        "(a) The taxpayer's calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "noun-fifth-token-intended-stricter",
        "(א) דוחות חישוב מפורטים של הנישום יוגשו בהתאם לסעיף 2.",
        "(a) The taxpayer's detailed calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "noun-14-spaces-intended-stricter",
        "(א)              דוחות חישוב יוגשו בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "noun-spaces-intended-stricter",
        "(א)                דוחות חישוב יוגשו בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "joint-control-intended-stricter",
        "(א) דוחות חישוב שיוגשו בהתאם לסעיף 2.",
        RELATIVE_E,
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "joint-intended-stricter",
        "(א) דוחות חישוב ש־יוגשו בהתאם לסעיף 2.",
        RELATIVE_E,
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "joint-ascii-intended-stricter",
        "(א) דוחות חישוב ש-יוגשו בהתאם לסעיף 2.",
        RELATIVE_E,
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "joint-spaced-intended-stricter",
        "(א) דוחות חישוב ש ־ יוגשו בהתאם לסעיף 2.",
        RELATIVE_E,
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "clause-head-intended-stricter",
        "(א) דוחות חישוב יוגשו בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "outline-8-intended-stricter",
        "(8) דוחות שיגישו בהתאם לסעיף 2.",
        "(8) Reports which they shall file in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "nominal-filing-form",
        "(א) ועדות המוסר דוחות חישוב בהתאם לסעיף 2.",
        "(a) The ethics committees reject calculation in accordance with section 2.",
        False,
        False,
        "(א) ועדות המוסר מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The ethics committees accept calculation in accordance with section 2.",
    ),
    (
        "nominal-bare-morality",
        "(א) ועדות מוסר דוחות חישוב בהתאם לסעיף 2.",
        "(a) Ethics committees reject calculation in accordance with section 2.",
        False,
        False,
        "(א) ועדות מוסר מקבלות חישוב בהתאם לסעיף 2.",
        "(a) Ethics committees accept calculation in accordance with section 2.",
    ),
    (
        "nominal-editor",
        "(א) הוראות העורך דוחות חישוב בהתאם לסעיף 2.",
        "(a) The editor's instructions reject calculation in accordance with section 2.",
        False,
        False,
        "(א) הוראות העורך מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The editor's instructions accept calculation in accordance with section 2.",
    ),
    (
        "nominal-recipient",
        "(א) טענות המקבל דוחות חישוב בהתאם לסעיף 2.",
        "(a) The recipient's claims reject calculation in accordance with section 2.",
        False,
        False,
        "(א) טענות המקבל מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The recipient's claims accept calculation in accordance with section 2.",
    ),
    (
        "authorities-present-intended-stricter",
        "(א) הרשויות מגישות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities file calculation reports in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "funds-present-intended-stricter",
        "(א) הקופות מכינות דוחות חישוב בהתאם לסעיף 2.",
        "(a) The funds prepare calculation reports in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "finite-passive-intended-stricter",
        "(א) יוגשו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "finite-active-intended-stricter",
        "(א) הנישום יגיש דוחות חישוב בהתאם לסעיף 2.",
        "(a) The taxpayer shall file calculation reports in accordance with section 2.",
        False,
        True,
        FILED_H,
        FILED_E,
    ),
    (
        "verb-first-positive",
        "(א) מקבלות הרשויות את החישובים שנמסרו בהתאם לסעיף 2.",
        "(a) The authorities accept the calculations that were submitted in accordance with section 2.",
        True,
        True,
        "(א) מקבלות הרשויות את החישובים שנמסרו בהתאם לסעיף 2.",
        "(a) The authorities accept the calculations that were submitted in accordance with section 2.",
    ),
    (
        "nominal-positive",
        "(א) ועדות המוסר מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The ethics committees accept calculation in accordance with section 2.",
        True,
        True,
        "(א) ועדות המוסר מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The ethics committees accept calculation in accordance with section 2.",
    ),
)


def _pair(hebrew: str, english: str) -> tuple[bool, bool]:
    branch = "8" if hebrew.startswith("(8)") else "a"
    return (
        _accepted(branch, "חסר סעיף 2", hebrew),
        _english_accepted(branch, "Missing section 2.", english),
    )


@pytest.mark.parametrize(
    ("hebrew", "english", "h_target", "e_target", "h_positive", "e_positive"),
    [row[1:] for row in ROUND_24_REPRODUCERS],
    ids=[row[0] for row in ROUND_24_REPRODUCERS],
)
def test_round_24_reproducers(
    hebrew, english, h_target, e_target, h_positive, e_positive
):
    assert _pair(hebrew, english) == (h_target, e_target)
    assert _pair(h_positive, e_positive) == (True, True)


INTENDED_STRICTER_ROUND_24 = tuple(
    (row[0], row[1], row[2])
    for row in ROUND_24_REPRODUCERS
    if row[3:5] == (False, True)
)


@pytest.mark.parametrize(
    ("hebrew", "english"),
    [row[1:] for row in INTENDED_STRICTER_ROUND_24],
    ids=[row[0] for row in INTENDED_STRICTER_ROUND_24],
)
def test_round_24_intended_stricter(hebrew, english):
    assert _pair(hebrew, english) == (False, True)
    assert guarded_snapshot_count(hebrew) == 0
