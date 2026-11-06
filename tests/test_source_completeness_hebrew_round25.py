"""Round 25: closed noun contexts and clause-bounded verb-first reports.

All 21 r13 replay rows, required additional rows and matched controls have
literal expectations from the round brief. Every (False, True) row is labelled
intended-stricter and measured against the external statute snapshot.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness import hebrew_source_text as hebrew
from tests.hebrew_reports_snapshot import guarded_snapshot_count
from tests.test_source_completeness_hebrew import _accepted, _english_accepted

# (id, Hebrew, faithful English, H target, E target, H positive, E positive).
ROUND_25_REPRODUCERS = (
    (
        "B1a",
        "(א) ועדות הערכה דוחות חישוב בהתאם לסעיף 2.",
        "(a) The assessment committees reject calculation in accordance with section 2.",
        False,
        False,
        "(א) ועדות הערכה מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The assessment committees accept calculation in accordance with section 2.",
    ),
    (
        "B1a-ctl",
        "(א) ועדות הערכה מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The assessment committees accept calculation in accordance with section 2.",
        True,
        True,
        "(א) ועדות הערכה מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The assessment committees accept calculation in accordance with section 2.",
    ),
    (
        "B1b",
        "(א) רשויות ניירות ערך דוחות חישוב בהתאם לסעיף 2.",
        "(a) The securities authorities reject calculation in accordance with section 2.",
        False,
        False,
        "(א) רשויות ניירות ערך מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The securities authorities accept calculation in accordance with section 2.",
    ),
    (
        "B1b-ctl",
        "(א) רשויות ניירות ערך מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The securities authorities accept calculation in accordance with section 2.",
        True,
        True,
        "(א) רשויות ניירות ערך מקבלות חישוב בהתאם לסעיף 2.",
        "(a) The securities authorities accept calculation in accordance with section 2.",
    ),
    (
        "B1c",
        "(א) בעת ההערכה דוחות הוועדות את החישוב בהתאם לסעיף 2.",
        "(a) At the assessment, the committees reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) בעת ההערכה מקבלות הוועדות את החישוב בהתאם לסעיף 2.",
        "(a) At the assessment, the committees accept the calculation in accordance with section 2.",
    ),
    (
        "B1c-ctl",
        "(א) בעת ההערכה מקבלות הוועדות את החישוב בהתאם לסעיף 2.",
        "(a) At the assessment, the committees accept the calculation in accordance with section 2.",
        True,
        True,
        "(א) בעת ההערכה מקבלות הוועדות את החישוב בהתאם לסעיף 2.",
        "(a) At the assessment, the committees accept the calculation in accordance with section 2.",
    ),
    (
        "B1c-comma",
        "(א) בעת ההערכה, דוחות הוועדות את החישוב בהתאם לסעיף 2.",
        "(a) At the assessment, the committees reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) בעת ההערכה, מקבלות הוועדות את החישוב בהתאם לסעיף 2.",
        "(a) At the assessment, the committees accept the calculation in accordance with section 2.",
    ),
    (
        "B1d",
        "(א) הבקשות יוגשו ודוחות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The applications shall be filed and the authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) הבקשות יוגשו ומקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The applications shall be filed and the authorities accept the calculation in accordance with section 2.",
    ),
    (
        "B1d-ctl",
        "(א) הבקשות יוגשו ומקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The applications shall be filed and the authorities accept the calculation in accordance with section 2.",
        True,
        True,
        "(א) הבקשות יוגשו ומקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The applications shall be filed and the authorities accept the calculation in accordance with section 2.",
    ),
    (
        "B1e",
        "(א) החברות שמניותיהן פורסמו דוחות כל חישוב בהתאם לסעיף 2.",
        "(a) The companies whose shares were published reject any calculation in accordance with section 2.",
        False,
        False,
        "(א) החברות שמניותיהן פורסמו מקבלות כל חישוב בהתאם לסעיף 2.",
        "(a) The companies whose shares were published accept any calculation in accordance with section 2.",
    ),
    (
        "B1e-ctl",
        "(א) החברות שמניותיהן פורסמו מקבלות כל חישוב בהתאם לסעיף 2.",
        "(a) The companies whose shares were published accept any calculation in accordance with section 2.",
        True,
        True,
        "(א) החברות שמניותיהן פורסמו מקבלות כל חישוב בהתאם לסעיף 2.",
        "(a) The companies whose shares were published accept any calculation in accordance with section 2.",
    ),
    (
        "B2-4w",
        "(א) דוחות רשויות המס המוסמכות בישראל את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות רשויות המס המוסמכות בישראל את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel accept the calculation in accordance with section 2.",
    ),
    (
        "B2-4w-ctl",
        "(א) מקבלות רשויות המס המוסמכות בישראל את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel accept the calculation in accordance with section 2.",
        True,
        True,
        "(א) מקבלות רשויות המס המוסמכות בישראל את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel accept the calculation in accordance with section 2.",
    ),
    (
        "B2-3w",
        "(א) דוחות רשויות המס המוסמכות את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות רשויות המס המוסמכות את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities accept the calculation in accordance with section 2.",
    ),
    (
        "B2-2w",
        "(א) דוחות רשויות המס את החישוב בהתאם לסעיף 2.",
        "(a) The tax authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות רשויות המס את החישוב בהתאם לסעיף 2.",
        "(a) The tax authorities accept the calculation in accordance with section 2.",
    ),
    (
        "B3.1-prefix-b-intended-stricter",
        "(א) הנישום יכלול בדוחות את החישוב בהתאם לסעיף 2.",
        "(a) The assessee shall include in the reports the calculation in accordance with section 2.",
        False,
        True,
        "(א) הנישום יכלול במסמכים את החישוב בהתאם לסעיף 2.",
        "(a) The assessee shall include in the documents the calculation in accordance with section 2.",
    ),
    (
        "B3.1-prefix-h-intended-stricter",
        "(א) יכללו הדוחות את החישוב בהתאם לסעיף 2.",
        "(a) The reports shall include the calculation in accordance with section 2.",
        False,
        True,
        "(א) יכללו המסמכים את החישוב בהתאם לסעיף 2.",
        "(a) The documents shall include the calculation in accordance with section 2.",
    ),
    (
        "B3.4-filing",
        "(א) המנהל רשאי לדרוש דוחות שיוגשו בהתאם לסעיף 2.",
        "(a) The director may require reports that shall be filed in accordance with section 2.",
        True,
        True,
        "(א) המנהל רשאי לדרוש דוחות שיוגשו בהתאם לסעיף 2.",
        "(a) The director may require reports that shall be filed in accordance with section 2.",
    ),
    (
        "N1-al",
        "(א) פקיד השומה רשאי לדרוש מהנישום דוחות על חישוב ההכנסה בהתאם לסעיף 2.",
        "(a) The assessing officer may require from the assessee reports on the calculation of income in accordance with section 2.",
        True,
        True,
        "(א) פקיד השומה רשאי לדרוש מהנישום דוחות על חישוב ההכנסה בהתאם לסעיף 2.",
        "(a) The assessing officer may require from the assessee reports on the calculation of income in accordance with section 2.",
    ),
    (
        "N1-shnatiyim",
        "(א) הנישום ימציא דוחות שנתיים על חיוב המס בהתאם לסעיף 2.",
        "(a) The assessee shall furnish annual reports on the tax liability in accordance with section 2.",
        True,
        True,
        "(א) הנישום ימציא דוחות שנתיים על חיוב המס בהתאם לסעיף 2.",
        "(a) The assessee shall furnish annual reports on the tax liability in accordance with section 2.",
    ),
    (
        "N1-ctl",
        "(א) יוגשו דוחות על החישוב בהתאם לסעיף 2.",
        "(a) Reports on the calculation shall be filed in accordance with section 2.",
        True,
        True,
        "(א) יוגשו דוחות על החישוב בהתאם לסעיף 2.",
        "(a) Reports on the calculation shall be filed in accordance with section 2.",
    ),
    (
        "B1-securities-exchange",
        "(א) לגבי ניירות ערך דוחות הבורסות את החישוב בהתאם לסעיף 2.",
        "(a) The securities exchanges reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) לגבי ניירות ערך מקבלות הבורסות את החישוב בהתאם לסעיף 2.",
        "(a) The securities exchanges accept the calculation in accordance with section 2.",
    ),
    (
        "B1-exchanges-fixed-payment",
        "(א) הבורסות לניירות ערך דוחות כל חישוב בהתאם לסעיף 2 ומחייבות תשלום קבוע.",
        "(a) The securities exchanges reject any calculation in accordance with section 2 and impose a fixed payment.",
        False,
        False,
        "(א) הבורסות לניירות ערך מקבלות כל חישוב בהתאם לסעיף 2 ומחייבות תשלום קבוע.",
        "(a) The securities exchanges accept any calculation in accordance with section 2 and impose a fixed payment.",
    ),
    (
        "B1-institutions-relative",
        "(א) המוסדות שהבקשות שלהם פורסמו דוחות כל חישוב בהתאם לסעיף 2.",
        "(a) The institutions whose applications were published reject any calculation in accordance with section 2.",
        False,
        False,
        "(א) המוסדות שהבקשות שלהם פורסמו מקבלות כל חישוב בהתאם לסעיף 2.",
        "(a) The institutions whose applications were published accept any calculation in accordance with section 2.",
    ),
    (
        "B2-5w",
        "(א) דוחות רשויות המס המוסמכות בישראל כיום את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel today reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות רשויות המס המוסמכות בישראל כיום את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel today accept the calculation in accordance with section 2.",
    ),
    (
        "B2-6w",
        "(א) דוחות רשויות המס המוסמכות בישראל כיום בלבד את החישוב בהתאם לסעיף 2.",
        "(a) Only the competent tax authorities in Israel today reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות רשויות המס המוסמכות בישראל כיום בלבד את החישוב בהתאם לסעיף 2.",
        "(a) Only the competent tax authorities in Israel today accept the calculation in accordance with section 2.",
    ),
    (
        "B2-8w",
        "(א) דוחות רשויות המס המוסמכות בישראל כיום ובכל שנה נוספת את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel today and in every additional year reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) מקבלות רשויות המס המוסמכות בישראל כיום ובכל שנה נוספת את החישוב בהתאם לסעיף 2.",
        "(a) The competent tax authorities in Israel today and in every additional year accept the calculation in accordance with section 2.",
    ),
    (
        "finite-passive-intended-stricter",
        "(א) יוגשו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        "(א) יוגשו מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents shall be filed in accordance with section 2.",
    ),
    (
        "finite-passive-definite-intended-stricter",
        "(א) יוגשו דוחות החישוב בהתאם לסעיף 2.",
        "(a) The calculation reports shall be filed in accordance with section 2.",
        False,
        True,
        "(א) יוגשו מסמכים החישוב בהתאם לסעיף 2.",
        "(a) The calculation documents shall be filed in accordance with section 2.",
    ),
    (
        "finite-active-intended-stricter",
        "(א) הנישום יגיש דוחות חישוב בהתאם לסעיף 2.",
        "(a) The taxpayer shall file calculation reports in accordance with section 2.",
        False,
        True,
        "(א) הנישום יגיש מסמכים חישוב בהתאם לסעיף 2.",
        "(a) The taxpayer shall file calculation documents in accordance with section 2.",
    ),
    (
        "finite-past-intended-stricter",
        "(א) הרשויות הגישו דוחות חישוב בהתאם לסעיף 2.",
        "(a) The authorities filed calculation reports in accordance with section 2.",
        False,
        True,
        "(א) הרשויות הגישו מסמכים חישוב בהתאם לסעיף 2.",
        "(a) The authorities filed calculation documents in accordance with section 2.",
    ),
    (
        "infinitive-file-intended-stricter",
        "(א) יש להגיש דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports must be filed in accordance with section 2.",
        False,
        True,
        "(א) יש להגיש מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents must be filed in accordance with section 2.",
    ),
    (
        "infinitive-prepare-intended-stricter",
        "(א) יש להכין דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports must be prepared in accordance with section 2.",
        False,
        True,
        "(א) יש להכין מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents must be prepared in accordance with section 2.",
    ),
    (
        "closed-preposition",
        "(א) הוראות לענין דוחות חישוב בהתאם לסעיף 2.",
        "(a) Provisions concerning calculation reports in accordance with section 2.",
        True,
        True,
        "(א) הוראות לענין דוחות חישוב בהתאם לסעיף 2.",
        "(a) Provisions concerning calculation reports in accordance with section 2.",
    ),
    (
        "closed-construct",
        "(א) הוראות להגשת דוחות חישוב בהתאם לסעיף 2.",
        "(a) Provisions for filing calculation reports in accordance with section 2.",
        True,
        True,
        "(א) הוראות להגשת דוחות חישוב בהתאם לסעיף 2.",
        "(a) Provisions for filing calculation reports in accordance with section 2.",
    ),
    (
        "B1-coordination-joint",
        "(א) הבקשות יוגשו ו־דוחות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The applications shall be filed and the authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) הבקשות יוגשו ו־מקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The applications shall be filed and the authorities accept the calculation in accordance with section 2.",
    ),
    (
        "B1-preposition-boundary",
        "(א) הוראות לענין ודוחות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) הוראות לענין ומקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities accept the calculation in accordance with section 2.",
    ),
    (
        "B1-relative-boundary",
        "(א) הוראות לענין ש־דוחות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities reject the calculation in accordance with section 2.",
        False,
        False,
        "(א) הוראות לענין ש־מקבלות הרשויות את החישוב בהתאם לסעיף 2.",
        "(a) The authorities accept the calculation in accordance with section 2.",
    ),
    (
        "prior-published-intended-stricter",
        "(א) פורסמו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports were published in accordance with section 2.",
        False,
        True,
        "(א) פורסמו מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents were published in accordance with section 2.",
    ),
    (
        "prior-received-relative-intended-stricter",
        "(א) שקיבלו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Those who received calculation reports in accordance with section 2.",
        False,
        True,
        "(א) שקיבלו מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Those who received calculation documents in accordance with section 2.",
    ),
    (
        "prior-received-future-intended-stricter",
        "(א) יתקבלו דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports shall be received in accordance with section 2.",
        False,
        True,
        "(א) יתקבלו מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents shall be received in accordance with section 2.",
    ),
    (
        "prior-prepared-future-intended-stricter",
        "(א) תוכנה דוחות חישוב בהתאם לסעיף 2.",
        "(a) Calculation reports shall be prepared in accordance with section 2.",
        False,
        True,
        "(א) תוכנה מסמכים חישוב בהתאם לסעיף 2.",
        "(a) Calculation documents shall be prepared in accordance with section 2.",
    ),
    (
        "prior-prepare-infinitive-intended-stricter",
        "(א) להכין דוחות חישוב בהתאם לסעיף 2.",
        "(a) To prepare calculation reports in accordance with section 2.",
        False,
        True,
        "(א) להכין מסמכים חישוב בהתאם לסעיף 2.",
        "(a) To prepare calculation documents in accordance with section 2.",
    ),
)


@pytest.mark.parametrize(
    "reason_h,reason_e",
    [
        ("חסר סעיף 2", "Missing section 2."),
        ("חסרות הוראות סעיף 2.", "Missing provisions of section 2."),
    ],
)
@pytest.mark.parametrize(
    "hebrew,english,h_target,e_target,h_positive,e_positive",
    [row[1:] for row in ROUND_25_REPRODUCERS],
    ids=[row[0] for row in ROUND_25_REPRODUCERS],
)
def test_round_25_reproducers(
    hebrew, english, h_target, e_target, h_positive, e_positive, reason_h, reason_e
):
    def pair(h, e):
        return _accepted("a", reason_h, h), _english_accepted("a", reason_e, e)

    assert pair(hebrew, english) == (h_target, e_target)
    assert pair(h_positive, e_positive) == (True, True)


INTENDED_STRICTER_ROUND_25 = tuple(
    row for row in ROUND_25_REPRODUCERS if row[3:5] == (False, True)
)


@pytest.mark.parametrize(
    "row",
    INTENDED_STRICTER_ROUND_25,
    ids=[row[0] for row in INTENDED_STRICTER_ROUND_25],
)
def test_round_25_intended_stricter(row):
    assert "intended-stricter" in row[0]
    assert guarded_snapshot_count(row[1]) == 0


@pytest.mark.parametrize("prefix", ("ו", "ש", "ו־", "ש ־ "))
def test_reports_subject_stops_at_next_prefixed_head(prefix):
    # Overlapping suffix lookaheads made repeated prefixed heads quadratic.
    # Only the final head owns the subject before its explicit object.
    text = hebrew.fold_marks(
        f"(א) דוחות הרשויות {prefix}דוחות רשויות המס המוסמכות את החישוב בהתאם לסעיף 2."
    )
    matches = [
        match
        for match in hebrew._NEGATIONS._regex.finditer(text)
        if match.group("reports_head") is not None
    ]
    assert len(matches) == 1
    assert matches[0].start() > text.index("הרשויות")
