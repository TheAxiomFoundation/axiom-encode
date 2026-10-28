"""The shared, finite Hebrew negator vocabulary and each form's prefix policy.

References: Gesenius–Kautzsch–Cowley, Hebrew Grammar (1910), §§109, 152
(negative particles, pronominal אין, פן, and jussive/cohortative אל):
https://en.wikisource.org/wiki/Gesenius%27_Hebrew_Grammar/152._Negative_Sentences
https://en.wikisource.org/wiki/Gesenius%27_Hebrew_Grammar/109._Use_of_the_Jussive
Also unfoldingWord Hebrew Grammar, Particle Negative (אל with imperfect,
jussive and cohortative; בלי/בל/בלא and לבלתי):
https://uhg.readthedocs.io/en/stable/particle_negative.html

These references identify the grammatical class, not a license to split every
word beginning with prefix letters. Short בל and אל take only clausal prefixes;
כבל is a cable. Lexical בלי, מבלי, בלא, ללא and באין have their own entries.
מעולם is negative polarity only with a following לא; מעולם העבודה is affirmative.
The legacy prohibition/exclusion verbs remain explicit below, with their
context restrictions. Statutory corpus controls pin both readings and exclusions.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass

from axiom_encode.harness.hebrew_prefixes import (
    HEBREW_PREFIX_LETTERS,
    NOT_ARTICLE_PREFIX,
    PREFIX_JOINT,
    PREFIX_NOT_WHOLE_WORD_JOINT,
    prefix_chain,
    prefix_word,
)

_H = "א-ת"
_PREFIX_LETTERS = HEBREW_PREFIX_LETTERS
# A conjunction, followed by a relative/subordinator. Bare prepositions and
# articles are excluded: מלא (full), כלא (prison), הלא (surely), מאין (whence).
ORDINARY_PREFIX_CHAINS = tuple(
    conjunction + subordinator
    for conjunction in ("", "ו")
    for subordinator in ("", "ש", "כש", "מש", "לכש", "בש", "ד")
)
GENERAL_PREFIX_CHAINS = tuple(
    "".join(chain)
    for length in range(4)
    for chain in itertools.product(HEBREW_PREFIX_LETTERS, repeat=length)
)
_NEGATION_PREFIX = (
    rf"(?:{PREFIX_NOT_WHOLE_WORD_JOINT}"
    rf"(?:{prefix_word('ו')})?"
    rf"(?:(?:{prefix_word('מ')}|{prefix_word('כ')}|{prefix_word('לכ')}|{prefix_word('ב')})?"
    rf"{prefix_word('ש')}|{prefix_word('ד')})?"
    r")?"
)
# Generic chains remain bounded at three; the independently grammatical
# conjunction/subordinator ולכש has four letters and remains admissible
# before both בלתי and the legacy lexical forms, without an unbounded chain.
_BILTI_PREFIX = rf"(?:{_NEGATION_PREFIX}|{prefix_chain(_PREFIX_LETTERS, 3)})"
# Keep the legacy greedy <=3 capture policy: relative-clause scope reads the
# captured prefix, so an ordinary-first union would change e.g. כשלסרב.
# The single explicit four-letter alternative restores the grammatical ולכש.
_LEXICAL_PREFIX = rf"(?:{prefix_chain(_PREFIX_LETTERS, 3)}|{prefix_word('ולכש')})"
_PREFIXED_BILTI = rf"{_BILTI_PREFIX}בלתי"
LEXICAL_PREFIX_CHAINS = tuple(
    dict.fromkeys((*GENERAL_PREFIX_CHAINS, *ORDINARY_PREFIX_CHAINS))
)
_NEGATION_GAP = rf"(?:\s|{PREFIX_JOINT})"
# These negative particles take exactly ORDINARY_PREFIX_CHAINS above. All
# pronominal forms are spelled out so none depends on a partial suffix list.
CLAUSAL_NEGATORS = (
    "לא",
    "אין",
    "אינני",
    "איני",
    "אינך",
    "אינו",
    "אינה",
    "איננו",
    "איננה",
    "אינכם",
    "אינכן",
    "אינם",
    "אינן",
    "טרם",
    "לאו",
    "פן",
    "לולא",
    "לולי",
    "אלמלא",
    "אלמלי",
    "אילולא",
    "אילולי",
)
_CLAUSAL_FORMS = "|".join(CLAUSAL_NEGATORS)
# All four imperfect person prefixes, not merely second person ת. With no
# vowels, תושב can also be a noun; this inherited conservative reading is
# explicitly intended in the differential tests.
_SHORT_FORMS = rf"(?:(?P<negative_bl>בל)(?!/\d)|{prefix_word('ל')}בל|אל(?={_NEGATION_GAP}+[תיאנ][{_H}]))"
_NEVER_FORM = (
    rf"מעולם(?=(?:{_NEGATION_GAP}+[\w]+){{0,6}}"
    rf"{_NEGATION_GAP}+{_NEGATION_PREFIX}לא(?![{_H}]))"
)
_WITHOUT_SIGNAL = (
    rf"(?:בלי|{prefix_word('ל')}לא|{prefix_word('מ')}בלי|{prefix_word('ב')}לא)"
)


@dataclass(frozen=True)
class NegatorFamily:
    """A closed family and its exact finite set of allowed prefix chains."""

    forms: tuple[str, ...]
    prefixes: tuple[str, ...]
    context: str = ""


# Legacy prohibition and exclusion forms are finite too. Their allowed prefix
# chains use the bounded lexical union above; אסר is separate because
# מ+אסר spells מאסר (imprisonment), not a prefixed negative verb.
LEGACY_LEXICAL_FORMS = (
    "אסור",
    "אסורה",
    "אסורים",
    "אסורות",
    "נאסר",
    "נאסרה",
    "נאסרו",
    "נאסרים",
    "נאסרות",
    "איסור",
    "מסרב",
    "מסרבת",
    "מסרבים",
    "מסרבות",
    "יסרב",
    "יסרבו",
    "תסרב",
    "תסרבנה",
    "תסרבו",
    "סרב",
    "סרבה",
    "סרבו",
    "סירב",
    "סירבה",
    "סירבו",
    "לסרב",
    "נבצר",
    "נבצרה",
    "מנוע",
    "מנועה",
    "מנועים",
    "מנועות",
    "כושל",
    "כושלת",
    "כושלים",
    "כושלות",
    "נכשל",
    "נכשלה",
    "נכשלו",
    "נכשלים",
    "נכשלות",
    "כשל",
    "כשלה",
    "כשלו",
    "יכשל",
    "יכשלו",
    "ייכשל",
    "ייכשלו",
    "תכשל",
    "תכשלנה",
    "תכשלו",
    "תיכשל",
    "תיכשלנה",
    "תיכשלו",
    "הוחרג",
    "הוחרגה",
    "הוחרגו",
    "מוחרג",
    "מוחרגת",
    "מוחרגים",
    "מוחרגות",
    "יוחרג",
    "יוחרגו",
    "תוחרג",
    "תוחרגנה",
    "נמנע",
    "נמנעה",
    "נמנעו",
    "נמנעים",
    "נמנעות",
    "יימנע",
    "יימנעו",
    "תימנע",
    "תימנענה",
    "דוחה",
    "דוחים",
    "דוחות",
    "שולל",
    "שוללת",
    "שוללים",
    "שוללות",
    "נשלל",
    "נשללה",
    "נשללו",
    "נשללים",
    "נשללות",
    "אוסר",
    "אוסרת",
    "אוסרים",
    "אוסרות",
    "יאסור",
    "תאסור",
    "יאסרו",
    "תאסרנה",
    "תאסורנה",
    "ייאסר",
    "ייאסרו",
    "תיאסר",
    "תיאסרנה",
    "מכחיש",
    "מכחישה",
    "מכחישים",
    "מכחישות",
    "הכחיש",
    "הכחישה",
    "הכחישו",
    "למעט",
    "מלבד",
    "להוציא",
    "אף",
)

NEGATOR_FAMILIES = (
    NegatorFamily(CLAUSAL_NEGATORS, ORDINARY_PREFIX_CHAINS),
    NegatorFamily(("בל", "לבל"), ORDINARY_PREFIX_CHAINS),
    NegatorFamily(("אל",), ORDINARY_PREFIX_CHAINS, "before ת/י/א/נ imperfect"),
    NegatorFamily(("בלתי",), LEXICAL_PREFIX_CHAINS),
    NegatorFamily(("בלי", "מבלי", "ללא", "בלא", "באין"), ORDINARY_PREFIX_CHAINS),
    NegatorFamily(("מעולם",), ORDINARY_PREFIX_CHAINS, "only before לא"),
    NegatorFamily(("אסר", "אסרה", "אסרו"), ORDINARY_PREFIX_CHAINS),
    NegatorFamily(("אי אפשר", "אי ניתן"), ORDINARY_PREFIX_CHAINS),
    NegatorFamily(("הלא",), ("",), "attributive לא before מנוצל"),
    NegatorFamily(
        ("זולת",),
        tuple(p for p in LEXICAL_PREFIX_CHAINS if not p.endswith("ה")),
        "no article: הזולת is another person",
    ),
    NegatorFamily(
        LEGACY_LEXICAL_FORMS,
        LEXICAL_PREFIX_CHAINS,
        "מנוע before מ; דוחות before את; אף with לא; להוציא ungoverned; זולת without article",
    ),
)


def build_negation_language(
    *,
    before: str,
    after: str,
    separator: str,
    governed_infinitive: str,
    except_infinitive: str,
) -> str:
    """Build the one pattern used by both reason and source guards."""

    _NOT_HEBREW_BEFORE, _NOT_HEBREW_AFTER = before, after
    _SEP = separator
    _GOVERNED_INFINITIVE, _EXCEPT_INFINITIVE = governed_infinitive, except_infinitive

    return (
        rf"{_NOT_HEBREW_BEFORE}(?:"
        rf"(?P<ordinary_prefix>{_NEGATION_PREFIX})"
        rf"(?:{_CLAUSAL_FORMS}|{_SHORT_FORMS}|{_NEVER_FORM}|אסר(?:ה|ו)?)"
        rf"|הלא(?={_NEGATION_GAP}+מנוצל{_NOT_HEBREW_AFTER})"
        rf"|(?P<bilti_prefix>{_BILTI_PREFIX})בלתי"
        rf"|{_NEGATION_PREFIX}(?:{_WITHOUT_SIGNAL}|{prefix_word('ב')}אין)"
        rf"|(?P<lexical_prefix>{_LEXICAL_PREFIX})"
        rf"(?:אסור(?:ה|ים|ות)?|נאסר(?:ה|ו|ים|ות)?|איסור|מסרב(?:ת|ים|ות)?"
        # Prevented (``מנוע מלפעול``), not the noun "engine" (``מס על מנוע``).
        rf"|מנוע(?:ה|ים|ות)?(?={_NEGATION_GAP}+{prefix_word('מ')}[{_H}])"
        # Refuses, will refuse, refused; unable (``נבצר מן המוסד``); fails,
        # will fail, failed, as English reads "refuses to", "unable to" and
        # "fails to".
        rf"|יסרב(?:ו)?|תסרב(?:נה|ו)?|סי?רב(?:ה|ו)?|{prefix_word('ל')}סרב|נבצר(?:ה)?"
        r"|כושל(?:ת|ים|ות)?|נכשל(?:ה|ו|ים|ות)?|כשל(?:ה|ו)?|י?יכשל(?:ו)?|תי?כשל(?:נה|ו)?"
        # Excluded, prevented, rejects, denies (not ``דוחות``, "reports").
        r"|הוחרג(?:ה|ו)?|מוחרג(?:ת|ים|ות)?|יוחרג(?:ו)?|תוחרג(?:נה)?"
        r"|נמנע(?:ה|ו|ים|ות)?|יימנע(?:ו)?|תימנע(?:נה)?|דוח(?:ה|ים)"
        r"|שולל(?:ת|ים|ות)?|נשלל(?:ה|ו|ים|ות)?"
        r"|אוסר(?:ת|ים|ות)?|יאסור|תאסור|יאסרו|תאסו?רנה|ייאסר(?:ו)?|תיאסר(?:נה)?"
        r"|מכחיש(?:ה|ים|ות)?|הכחיש(?:ה|ו)?"
        # ``דוחות את הטענה`` ("reject the claim"), not ``דוחות`` ("reports").
        rf"|דוחות(?=\s+את{_NOT_HEBREW_AFTER})"
        # Except, other than: ``להוציא`` but not the infinitive after a modal
        # (``יש להוציא רישיון``, "a license must be issued"), and ``זולת`` but not
        # the noun ``הזולת`` ("another person").
        rf"|{prefix_word('ל')}מעט|{prefix_word('מ')}לבד"
        rf"|(?P<exceptional_infinitive>{_GOVERNED_INFINITIVE}{_EXCEPT_INFINITIVE})|{NOT_ARTICLE_PREFIX}זולת"
        # אף is negative polarity with a following לא ("not even"), rather than
        # the affirmative connective על אף / אף כי ("despite" / "although").
        rf"|(?P<negative_af>אף)(?=(?:{_NEGATION_GAP}+\w+){{0,6}}"
        rf"{_NEGATION_GAP}+{_NEGATION_PREFIX}לא{_NOT_HEBREW_AFTER}))"
        rf"{_NOT_HEBREW_AFTER}"
        rf"){_NOT_HEBREW_AFTER}"
        rf"|{_NOT_HEBREW_BEFORE}{_NEGATION_PREFIX}אי{_SEP}(?:אפשר|ניתן){_NOT_HEBREW_AFTER}"
        rf"|{_NOT_HEBREW_BEFORE}(?:{prefix_word('ו')})?"
        rf"(?:פרט{_SEP}?{prefix_word('ל')}|(?P<exclusion_huts>חוץ){_SEP}?{prefix_word('מ')})\s*(?=[^\W\d_])"
    )


# These are grammatical homographs in the receipt-pinned statute snapshot.
# Filtering reads the actual neighboring word rather than widening the finite
# prefix inventory or relying on a fixed-width whitespace lookbehind.
_ISSUING_GOVERNORS = frozenset(
    ("הורשו", "שהורשו", "הרשה", "להרשות", "רשאית", "שרשאית", "שלא")
)
_FOREIGN_HEADS = frozenset(
    (
        "מדינת",
        "מדינה",
        "תושב",
        "תושבי",
        "תושבת",
        "גופי",
        "גופים",
        "מטבע",
        "עיתונאי",
        "הכנסות",
        "מסי",
    )
)


def _previous_word(text: str, end: int) -> str:
    while end and text[end - 1].isspace():
        end -= 1
    start = end
    while start and "א" <= text[start - 1] <= "ת":
        start -= 1
    return text[start:end]


def _genuine_context(match: re.Match[str]) -> bool:
    text = match.string
    if match.group("exceptional_infinitive"):
        previous = _previous_word(text, match.start("exceptional_infinitive"))
        if previous in _ISSUING_GOVERNORS:
            return False
        if previous == "קופה":
            start = match.start("exceptional_infinitive")
            while start and text[start - 1].isspace():
                start -= 1
            if _previous_word(text, start - len(previous)) in {"רשאית", "שרשאית"}:
                return False
    if match.group("negative_af"):
        previous = _previous_word(text, match.start("negative_af"))
        if previous.endswith("על") and all(
            c in HEBREW_PREFIX_LETTERS for c in previous[:-2]
        ):
            return False
        # אף אם/כי/שלא (even if/although), and אף בשונה (even contrary to),
        # remain concessive even when a later independent לא is present.
        if re.match(
            r"\s+(?:אם|כי|שלא|שאינו|שאינה|שאינם|שאינן|שהוא|שהיא|שהם|שהן|בשונה)(?![א-ת])",
            text[match.end("negative_af") : match.end("negative_af") + 16],
        ):
            return False
    if match.group("exclusion_huts"):
        previous = _previous_word(text, match.start("exclusion_huts"))
        if any(
            previous.endswith(head)
            and len(previous) - len(head) <= 3
            and all(c in HEBREW_PREFIX_LETTERS for c in previous[: -len(head)])
            for head in _FOREIGN_HEADS
        ):
            return False
    return True


class NegationPattern:
    """A shared regex with statutory homograph exclusions in every reader."""

    def __init__(self, language: str) -> None:
        self._regex = re.compile(language)
        self.pattern = self._regex.pattern

    def finditer(
        self, text: str, pos: int = 0, endpos: int | None = None
    ) -> Iterator[re.Match[str]]:
        for match in self._regex.finditer(
            text, pos, len(text) if endpos is None else endpos
        ):
            if _genuine_context(match):
                yield match

    def search(
        self, text: str, pos: int = 0, endpos: int | None = None
    ) -> re.Match[str] | None:
        return next(self.finditer(text, pos, endpos), None)

    def fullmatch(
        self, text: str, pos: int = 0, endpos: int | None = None
    ) -> re.Match[str] | None:
        match = self._regex.fullmatch(
            text, pos, len(text) if endpos is None else endpos
        )
        return match if match and _genuine_context(match) else None
