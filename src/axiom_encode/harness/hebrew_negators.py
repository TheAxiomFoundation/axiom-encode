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
כבל is a cable. Lexical בלי, מבלי, בלא, ללא and באין have their own entries,
as do בהיעדר, בהעדר and בחוסר (ABSENCE_NOUNS below).
מעולם is negative polarity only with a following לא; מעולם העבודה is affirmative.
The legacy prohibition/exclusion verbs remain explicit below, with their
context restrictions. Statutory corpus controls pin both readings and exclusions.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from axiom_encode.harness.hebrew_prefixes import (
    HEBREW_PREFIX_LETTERS,
    HORIZONTAL_SPACES,
    NOT_ARTICLE_PREFIX,
    OTHER_SCRIPT_LETTER,
    PREFIX_JOINT,
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
    rf"(?:(?:{prefix_word('ו')})?"
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
# "Without" is also ב with an absence noun in construct, as באין is: בהיעדר
# and its defective spelling בהעדר ("in the absence of"), and בחוסר ("for
# want of"). In the statute snapshot, בהיעדר עדות לגרורות and בהעדר רופא.
# Only that construct reads "without". The bare nouns and their causal uses
# (עקב היעדר, "due to the absence of"; מחוסר, "for lack of") are nouns, as
# English "absence" and "lack" are, and a pronoun suffix (בהעדרה, "in its
# absence") leaves no object to negate. בחסר is no spelling of בחוסר in the
# statutes: ביתר או בחסר is "overstated or understated", מכירה בחסר a short
# sale. Like באין, these negate; they are no missing-dependency signal.
ABSENCE_NOUNS = ("היעדר", "העדר", "חוסר")
_ABSENCE_FORMS = "|".join(ABSENCE_NOUNS)


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
    NegatorFamily(
        tuple("ב" + noun for noun in ABSENCE_NOUNS),
        ORDINARY_PREFIX_CHAINS,
        "ב with an absence noun in construct: without",
    ),
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


# Modals and purpose words that make להוציא the infinitive "to issue" (יש
# להוציא רישיון, "a license must be issued") rather than "except".
INFINITIVE_GOVERNORS = frozenset(
    "יש צריך צריכה צריכים צריכות חייב חייבת חייבים חייבות רשאי רשאית רשאים "
    "רשאיות מוסמך מוסמכת מוסמכים מוסמכות יכול יכולה יכולים יכולות ניתן אפשר "
    "כדי בכדי מנת עליו עליה עליהם עליהן".split()
)
_LONGEST_GOVERNOR = max(map(len, INFINITIVE_GOVERNORS))
# Geresh and gershayim stand inside a Hebrew word (ג׳וגולר, מע״מ). In the
# statute snapshot neither ever opens a token.
_WORD_INTERNAL_MARKS = "׳״"
_OTHER_SCRIPT_LETTER = re.compile(OTHER_SCRIPT_LETTER)


def _continues_word(char: str) -> bool:
    """Whether ``char`` belongs to the word that follows it.

    A letter or digit of any script, connector punctuation (``_``), a
    combining mark, a format character (a joiner the fold has not removed)
    or a geresh mark. A space, a line break, a dash or a quotation mark does
    not; a dash after a full word is punctuation (hebrew_prefixes).
    """

    category = unicodedata.category(char)
    return (
        char.isalnum()
        or category[0] == "M"
        or category in ("Pc", "Cf")
        or char in _WORD_INTERNAL_MARKS
    )


def starts_token(text: str, begin: int) -> bool:
    """Whether the Hebrew word at ``begin`` starts a token as written.

    Nothing to its left may continue it: ``_יש``, ``1יש`` and ``Xיש`` are one
    token each, not the governor יש. Readers of negations read a copy with a
    space at each script junction (hebrew_source_text._split_scripts), where
    ``Xיש`` reads ``X יש``. After a letter of another script and one space,
    the text as written may have held no space, so that word is not read as
    a token start either; this is the stricter reading. Constant time.
    """

    if not begin:
        return True
    before = text[begin - 1]
    if _continues_word(before):
        return False
    return not (
        before == " "
        and begin > 1
        and _OTHER_SCRIPT_LETTER.match(text, begin - 2) is not None
    )


def governs_infinitive(text: str, start: int) -> bool:
    """Whether a whole governor word and horizontal whitespace end at ``start``.

    Any run of horizontal whitespace separates a governor from its infinitive
    (יש  להוציא, יש\tלהוציא); a dash after the full word, a line break or a
    prefix does not. The governor is a complete token (starts_token), so the
    end of an identifier (``_יש``, ``1יש``) governs nothing. Readers ask only
    where להוציא matched; the scan reads back over that whitespace run, at
    most one governor's length and one character before it.
    """

    end = start
    while end and text[end - 1] in HORIZONTAL_SPACES:
        end -= 1
    if end == start:
        return False
    begin = end
    while begin and end - begin <= _LONGEST_GOVERNOR and "א" <= text[begin - 1] <= "ת":
        begin -= 1
    return text[begin:end] in INFINITIVE_GOVERNORS and starts_token(text, begin)


def build_negation_language(
    *,
    before: str,
    after: str,
    separator: str,
    except_infinitive: str,
) -> str:
    """Build the one pattern used by both reason and source guards."""

    _NOT_HEBREW_BEFORE, _NOT_HEBREW_AFTER = before, after
    _SEP = separator
    _EXCEPT_INFINITIVE = except_infinitive

    return (
        rf"{_NOT_HEBREW_BEFORE}(?:"
        rf"(?P<ordinary_prefix>{_NEGATION_PREFIX})"
        rf"(?:{_CLAUSAL_FORMS}|{_SHORT_FORMS}|{_NEVER_FORM}|אסר(?:ה|ו)?)"
        rf"|הלא(?={_NEGATION_GAP}+מנוצל{_NOT_HEBREW_AFTER})"
        rf"|(?P<bilti_prefix>{_BILTI_PREFIX})בלתי"
        rf"|{_NEGATION_PREFIX}(?:{_WITHOUT_SIGNAL}|{prefix_word('ב')}(?:אין|{_ABSENCE_FORMS}))"
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
        # (``יש להוציא רישיון``, "a license must be issued"; governs_infinitive
        # filters it), and ``זולת`` but not the noun ``הזולת`` ("another person").
        rf"|{prefix_word('ל')}מעט|{prefix_word('מ')}לבד"
        rf"|(?P<exceptional_infinitive>{_EXCEPT_INFINITIVE})|{NOT_ARTICLE_PREFIX}זולת"
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
    """The Hebrew word before ``end``; empty unless it is a whole token.

    A context word ends an exception only as a complete token
    (starts_token): ``_הורשו להוציא`` and ``1מדינת חוץ מ`` keep their exclusion.
    """

    while end and text[end - 1].isspace():
        end -= 1
    start = end
    while start and "א" <= text[start - 1] <= "ת":
        start -= 1
    return text[start:end] if starts_token(text, start) else ""


def _genuine_context(match: re.Match[str]) -> bool:
    text = match.string
    if match.group("exceptional_infinitive"):
        if governs_infinitive(text, match.start("exceptional_infinitive")):
            return False
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


# --- Source-only negators (#1882, round 19) ---------------------------------
#
# The reason reader keeps the shared language above: there ``חסר סעיף 2`` is
# the missing-dependency signal. A source reads more words as negating its
# link, as English's source guard reads ``without calculation``, ``prevented``
# and ``denied`` (``_SOURCE_LINK_NEGATION_LANGUAGE``). Two rules add them, and
# only to the source reader (hebrew_source_text._NEGATIONS).
#
# 1. A form the statute snapshot holds in no other sense reads clause-wide, as
#    לא does, with the same prefix chains. Each form below is 0/0 in the
#    snapshot's bodies and headings under every ordinary prefix chain, except
#    ``הוצאו מכלל``, whose one use (National Insurance Law section 75, "classes
#    of persons excluded from the self-employed") is the same exclusion.
SOURCE_CLAUSE_NEGATORS = (
    # Blocked, frustrated (prevented).
    "נחסם",
    "נחסמה",
    "נחסמו",
    "סוכל",
    "סוכלה",
    "סוכלו",
    "מסוכל",
    "מסוכלת",
    "מסוכלים",
    "מסוכלות",
    "מסכל",
    "מסכלת",
    "מסכלים",
    "מסכלות",
    # Prevented, the form the legacy נמנע family lacks.
    "נמנעת",
    # Negated, denied, refuses, disowns.
    "שלול",
    "שלולה",
    "שלולים",
    "שלולות",
    "הוכחש",
    "הוכחשה",
    "הוכחשו",
    "ממאן",
    "ממאנת",
    "ממאנים",
    "ממאנות",
    "מתכחש",
    "מתכחשת",
    "מתכחשים",
    "מתכחשות",
    "התכחש",
    "התכחשה",
    "התכחשו",
    "יתכחש",
    "תתכחש",
    "יתכחשו",
    "תתכחשנה",
    # Deprived of: without (משולל also reads as מ + the legacy שולל).
    "משולל",
    "משוללת",
    "משוללות",
)
# Clause-wide forms that take their complement: disclaims (מתנער מ),
# excluded from (מוצא מכלל), and the denied proposition (שקר ש, "false that").
SOURCE_CLAUSE_PHRASES = (
    *(f"{form} מ" for form in ("מתנער", "מתנערת", "מתנערים", "מתנערות")),
    *(
        f"{form} מכלל"
        for form in ("מוצא", "מוצאת", "מוצאים", "מוצאות", "הוצא", "הוצאה", "הוצאו")
    ),
    *(f"{form} ש" for form in ("שקר", "שקרי", "שגוי")),
)

# 2. A form with a statutory homograph (נעדר, a missing person; פטור, exempt
#    from tax; במקום, at a place; כופר, a monetary settlement) reads only over
#    a computation complement, as English ``without`` reads only before
#    ``determin*``, ``comput*``, ``calculat*``, ``appl*``, ``impos*`` and
#    ``aris*``: head, optional quantifier, optional article, computation head.
#    The computation heads are the Hebrew of those six English stems, and the
#    stems themselves for a mixed clause (``נטול calculation``). ``תשלום`` is
#    no computation head: ``אי תשלום בהתאם לסעיף 2`` is "non-payment under".
SOURCE_PRIVATIVE_HEADS = (
    # Without, lacking: the construct adjectives, agreeing in gender and
    # number. The absolute plurals take ב: ``נעדרים בחישוב``.
    "נטול",
    "נטולת",
    "נטולי",
    "נטולות",
    "נעדר",
    "נעדרת",
    "נעדרי",
    "נעדרות",
    "חסר",
    "חסרה",
    "חסרת",
    "חסרי",
    "חסרות",
    # מחוסר alone is also causal ("for lack of"): see _LACK_CONTEXT.
    "מחוסר",
    "מחוסרת",
    "מחוסרי",
    "מחוסרות",
    "משוללי",
    # Literary "without": בלעדי, באפס.
    "בלעדי",
    "באפס",
    # Blocked, prevents, prevention of, rejected, reject.
    "חסום",
    "חסומה",
    "חסומים",
    "חסומות",
    "מונע",
    "מונעת",
    "מונעים",
    "מונעות",
    "מניעת",
    "נדחה",
    "נדחתה",
    "נדחו",
    "דוחות",
    # Non- over an action noun (בלתי already negates clause-wide).
    "אי",
    # Instead of: "rather than".
    "במקום",
    "חלף",
    "תחת",
)
# Heads whose computation complement follows ``מ`` or ``מן``: exempt from,
# other than, except.
SOURCE_FROM_HEADS = (
    "פטור",
    "פטורה",
    "פטורים",
    "פטורות",
    "שונה",
    "שונים",
    "שונות",
    "בשונה",
    "להבדיל",
    "לבד",
)
# The exemption's construct forms take the complement directly: ``פטורי חישוב``.
SOURCE_CONSTRUCT_HEADS = ("פטורי", "פטורת", "פטורות")
# A predicate after its computation subject, with its ``מן`` complement:
# ``החישוב נפקד מן התשלום``, "the calculation is absent from the payment".
SOURCE_ABSENT_FROM_PREDICATES = (
    "נפקד",
    "נפקדת",
    "נפקדים",
    "נפקדות",
    "נעדר",
    "נעדרת",
    "נעדרים",
    "נעדרות",
    "חסר",
    "חסרה",
    "חסרים",
    "חסרות",
)
SOURCE_QUANTIFIERS = ("כל", "שום", "אותו", "אותה", "אותם", "אותן")
ABSENCE_SUFFIXES = ("ו", "ה", "ם", "ן")
# Denies a claim: ``כופר בטענה ש``, ``כופר בכך ש``; ``כופר כסף`` is money.
DENIAL_HEADS = ("כופר", "כופרת", "כופרים", "כופרות")
CLAIM_NOUNS = (
    "טענה",
    "טענות",
    "דיווח",
    "דיווחים",
    "הנחה",
    "אמירה",
    "גרסה",
)
# Copulas before the privative adjective (``יהיה מחוסר חישוב``).
COPULAS = frozenset(
    "היה היתה הייתה היו יהיה תהיה יהיו תהיינה להיות הוא היא הם הן הינו הינה "
    "הינם הינן הנו הנה הנם הנן".split()
)

# The computation heads, by family. Nouns take their construct and pronoun
# suffixes; infinitives and the relative future read after verbs of refusal
# and after ``במקום``. ``שיקום`` and ``שיקומו`` are left out: in the statutes
# they are "rehabilitation", never ש with ``יקום`` ("that it arise").
_COMPUTATION_NOUNS = (
    r"קביע(?:ה|ת|ות|ותי|תו|תה|תם|תן)|היקבעות"
    r"|חישוב(?:ים|י|ו|ה|ם|ן)?|עיבוד(?:ו)?\s+(?:ה)?חישובי"
    r"|החל(?:ה|ת|ות|תו|תה|תם|תן)|יישום(?:ים|י|ו|ה|ם|ן)?"
    r"|תחול(?:ה|ת|תו|תה|תם|תן)"
    r"|הטל(?:ה|ת|ות|תו|תה|תם|תן)|השת(?:ה|ת|תו|תה|תם|תן)|חיוב(?:ים|י|ו|ה|ם|ן)?"
    r"|היווצרות(?:ו|ה|ם|ן)?|התהוות(?:ו|ה|ם|ן)?|צמיח(?:ה|ת|תו|תה|תם|תן)"
)
_COMPUTATION_VERBS = (
    r"להיקבע|לחשב|להחיל|ליישם|להטיל|לחייב|להיווצר|לצמוח|לחול|להשית|לקום"
    r"|להיות\s+מוחל(?:ת|ים|ות)?"
    r"|ש(?:ייקבע(?:ו)?|תיקבע|יחושב(?:ו)?|תחושב|יוחל(?:ו)?|תוחל|ייושם(?:ו)?"
    r"|תייושם|יוטל(?:ו)?|תוטל|יחויב(?:ו)?|תחויב|ייווצר(?:ו)?|תיווצר|יצמח(?:ו)?"
    r"|תצמח|יחול(?:ו)?|תחול|יושת(?:ו)?|תושת|תקום)"
)
# English's six stems, for a mixed clause; English reads them with \b.
_ENGLISH_COMPUTATION = r"(?i:(?:determin|comput|calculat|appl|impos|aris)[a-z]*\b)"


def _alternation_of(words: Iterable[str]) -> str:
    return "|".join(map(re.escape, sorted(words, key=len, reverse=True)))


def build_source_negation_language(
    *,
    before: str,
    after: str,
    separator: str,
    except_infinitive: str,
) -> str:
    """The reason language, and the source-only forms after it.

    Every source-only alternative shares one ``source_prefix`` group, the
    ordinary prefix chains לא takes, so ``_is_relative`` reads ``שנטול`` as a
    relative clause. A head matches alone; its complement is a lookahead, so
    the match ends before the computation head, and the link after it.
    """

    gap = separator
    article = rf"(?:{prefix_word('ה')})?"
    quantifier = rf"(?:(?:{_alternation_of(SOURCE_QUANTIFIERS)}){after}{gap})?"
    noun = rf"(?:{_COMPUTATION_NOUNS}){after}"

    def computation(definite: str = article) -> str:
        return (
            rf"{quantifier}(?:{definite}(?:{noun}|(?:{_COMPUTATION_VERBS}){after})"
            rf"|{_ENGLISH_COMPUTATION})"
        )

    from_complement = (
        rf"(?:{prefix_word('מ')}|מן{after}{gap}){quantifier}{article}{noun}"
        rf"|מן{after}{gap}{quantifier}{_ENGLISH_COMPUTATION}"
    )
    clause_phrases = "|".join(
        re.escape(head)
        + after
        + gap
        + re.escape(tail)
        + (after if len(tail) > 1 else rf"(?=[{_H}])")
        for head, tail in (phrase.split() for phrase in SOURCE_CLAUSE_PHRASES)
    )
    privative = _alternation_of(
        head for head in SOURCE_PRIVATIVE_HEADS if head != "מחוסר"
    )
    claim = (
        rf"{prefix_word('ב')}(?:{prefix_word('ה')})?(?:{_alternation_of(CLAIM_NOUNS)})"
        rf"{after}{gap}(?:{prefix_word('ש')}(?=[{_H}])|כי{after})"
        rf"|בכך{after}{gap}(?:{prefix_word('ש')}(?=[{_H}])|כי{after})"
    )
    source_only = (
        rf"{before}(?P<source_prefix>{_NEGATION_PREFIX})(?:"
        rf"(?:{_alternation_of(SOURCE_CLAUSE_NEGATORS)}){after}"
        rf"|{clause_phrases}"
        rf"|(?P<guarded>)(?:"
        rf"(?:{privative}){after}(?={gap}{computation()})"
        rf"|(?P<lack>מחוסר){after}(?={gap}{computation(rf'(?P<lack_article>{article})')})"
        rf"|(?:נעדרים|חסרים){after}(?={gap}{prefix_word('ב')}{article}{noun})"
        rf"|(?:{_alternation_of(SOURCE_FROM_HEADS)}){after}(?={gap}?(?:{from_complement}))"
        rf"|(?:{_alternation_of(SOURCE_CONSTRUCT_HEADS)}){after}(?={gap}{computation()})"
        rf"|{prefix_word('ב')}(?:היעדר|העדר|חוסר)(?:{'|'.join(ABSENCE_SUFFIXES)}){after}"
        rf"(?={gap}(?:המוחלט{after}{gap})?של{after}{gap}{computation()})"
        rf"|(?:{_alternation_of(DENIAL_HEADS)}){after}(?={gap}(?:{claim}))"
        rf"|{article}(?:{_COMPUTATION_NOUNS}|{_ENGLISH_COMPUTATION}){after}{gap}"
        rf"(?:{_alternation_of(SOURCE_ABSENT_FROM_PREDICATES)}){after}"
        rf"(?={gap}?(?:{prefix_word('מ')}(?=[{_H}])|מן{after}))"
        r"))"
    )
    language = build_negation_language(
        before=before,
        after=after,
        separator=separator,
        except_infinitive=except_infinitive,
    )
    return rf"(?:{language})|(?:{source_only})"


# A concessive preposition takes the guarded head as its object: ``למרות אי
# קביעתו מראש, התשלום מחושב בהתאם לסעיף 2`` concedes the non-determination
# ("despite it"), and the payment is still computed under section 2.
CONCESSIVE_PREPOSITIONS = frozenset(("למרות", "ולמרות", "חרף", "וחרף", "אף", "ואף"))


def _concedes(match: re.Match[str]) -> bool:
    """Whether a concessive preposition governs a guarded source head."""

    if match.group("guarded") is None:
        return False
    return _previous_word(match.string, match.start()) in CONCESSIVE_PREPOSITIONS


def _lack_is_causal(match: re.Match[str]) -> bool:
    """Whether a bare ``מחוסר`` is causal ("for lack of"), not "lacking".

    Uninflected ``מחוסר`` before an indefinite complement, with no copula
    or relative before it, is the causal noun (``התשלום יבוצע מחוסר חישוב``,
    "made for lack of calculation"), as round 18 pinned it. After a copula
    (``יהיה מחוסר חישוב``), in a relative clause (``שמחוסר``) or before a
    definite complement (``מחוסר החישוב``) it is the adjective, "without".
    """

    if match.group("lack") is None:
        return False
    if set(match.group("source_prefix") or "") & {"ש", "ד"}:
        return False
    if match.group("lack_article"):
        return False
    previous = _previous_word(match.string, match.start())
    for prefix in ("", "ו", "ש", "וש"):
        if previous.startswith(prefix) and previous[len(prefix) :] in COPULAS:
            return False
    return True


class NegationPattern:
    """A shared regex with statutory homograph exclusions in every reader."""

    def __init__(self, language: str) -> None:
        self._regex = re.compile(language)
        self.pattern = self._regex.pattern
        self._source = "lack" in self._regex.groupindex

    def _kept(self, match: re.Match[str]) -> bool:
        return _genuine_context(match) and not (
            self._source and (_lack_is_causal(match) or _concedes(match))
        )

    def finditer(
        self, text: str, pos: int = 0, endpos: int | None = None
    ) -> Iterator[re.Match[str]]:
        for match in self._regex.finditer(
            text, pos, len(text) if endpos is None else endpos
        ):
            if self._kept(match):
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
        return match if match and self._kept(match) else None
