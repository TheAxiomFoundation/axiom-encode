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
from dataclasses import dataclass, field

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


# --- Source-only grammatical paradigms (#1882, round 20) -------------------
SOURCE_PREFIX_LANGUAGE = _NEGATION_PREFIX
#
# These are grammatical stems, not a dictionary. All endings live in this
# generator; callers select a family or a grammatical slot. There are ten
# possessive slots. Singular-noun second-person slots coincide without
# niqqud; full plural spelling distinguishes them.
# Source-only expansion leaves build_negation_language (the reason reader)
# unchanged. No third-party dictionary data is used.
_FINAL_TO_MEDIAL = str.maketrans("םןךףץ", "מנכפצ")
_MEDIAL_TO_FINAL = str.maketrans("מנכפצ", "םןךףץ")
_POSSESSIVE_ENDINGS = (
    ("י", "ך", "ך", "ו", "ה", "נו", "כם", "כן", "ם", "ן"),
    ("יי", "יך", "ייך", "יו", "יה", "ינו", "יכם", "יכן", "יהם", "יהן"),
)
# Full unpointed writing distinguishes plural 1s (-ay) and 2fs (-ayikh)
# from the construct/2ms endings. Pointed spelling writes one י in those
# slots; after fold_marks removes niqqud those grammatical spellings remain
# valid (טַעֲנוֹתַי -> טענותי). These are suffix orthographies, not new stems.
_VOCALIZED_PLURAL_ENDINGS = tuple(
    ending[1:] if slot in (0, 2) else ending
    for slot, ending in enumerate(_POSSESSIVE_ENDINGS[1])
)
_THIRD_PERSON_POSSESSIVE_SLOTS = (3, 4, 8, 9)


def _joined(stem: str, ending: str = "") -> str:
    """Use a medial letter before an ending, a final letter at word end."""

    if not stem:
        return ending
    return (
        stem[:-1]
        + stem[-1].translate(_FINAL_TO_MEDIAL if ending else _MEDIAL_TO_FINAL)
        + ending
    )


def _unique(words: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(words))


@dataclass(frozen=True)
class NounParadigm:
    name: str
    absolute: tuple[str, str]
    construct: tuple[str, str]
    singular_possessive: tuple[str, ...]
    plural_possessive: tuple[str, ...]
    plural_possessive_variants: tuple[str, ...]

    @property
    def forms(self) -> tuple[str, ...]:
        return _unique(
            (
                *self.absolute,
                *self.construct,
                *self.singular_possessive,
                *self.plural_possessive,
                *self.plural_possessive_variants,
            )
        )


def noun_paradigm(singular: str, plural: str) -> NounParadigm:
    """Absolute/construct singular/plural and all ten possessive slots."""

    singular_construct = singular[:-1] + "ת" if singular.endswith("ה") else singular
    plural_construct = plural[:-2] + "י" if plural.endswith("ים") else plural
    plural_stem = plural[:-2] if plural.endswith("ים") else plural
    return NounParadigm(
        singular,
        (singular, plural),
        (singular_construct, plural_construct),
        tuple(_joined(singular_construct, suffix) for suffix in _POSSESSIVE_ENDINGS[0]),
        tuple(_joined(plural_stem, suffix) for suffix in _POSSESSIVE_ENDINGS[1]),
        tuple(_joined(plural_stem, suffix) for suffix in _VOCALIZED_PLURAL_ENDINGS),
    )


@dataclass(frozen=True)
class AdjectiveParadigm:
    name: str
    absolute: tuple[str, str, str, str]
    construct: tuple[str, str, str, str]
    # An adjective has no tense or infinitive: those belong to a verb family.
    missing_slots: dict[str, str] = field(
        default_factory=lambda: {
            "past": "adjective, not a finite verb",
            "future": "adjective, not a finite verb",
            "infinitive": "adjective, not a finite verb",
        }
    )

    @property
    def forms(self) -> tuple[str, ...]:
        return _unique((*self.absolute, *self.construct))


def adjective_paradigm(base: str, *, feminine: str = "he") -> AdjectiveParadigm:
    """Four agreement slots, and their construct forms."""

    # י adjectives retain their stem י before feminine and plural endings.
    feminine_ending = "ת" if feminine == "tav" else "ה"
    if base.endswith("ה"):
        stem = base[:-1]
        return AdjectiveParadigm(
            base,
            (base, base, _joined(stem, "ים"), _joined(stem, "ות")),
            (base, _joined(stem, "ת"), _joined(stem, "י"), _joined(stem, "ות")),
        )
    return AdjectiveParadigm(
        base,
        (
            base,
            _joined(base, feminine_ending),
            _joined(base, "ים"),
            _joined(base, "ות"),
        ),
        (base, _joined(base, "ת"), _joined(base, "י"), _joined(base, "ות")),
    )


@dataclass(frozen=True)
class VerbParadigm:
    name: str
    past: tuple[str, str, str]
    present: tuple[str, str, str, str]
    future: tuple[str, str, str, str]
    infinitive: tuple[str, ...]
    missing_slots: dict[str, str]

    @property
    def forms(self) -> tuple[str, ...]:
        return _unique((*self.past, *self.present, *self.future, *self.infinitive))


def verb_paradigm(
    past: str,
    present: str,
    future: str,
    *,
    feminine: str = "tav",
    past_stem: str | None = None,
    future_plural: str | None = None,
    future_feminine: str | None = None,
    present_feminine_stem: str | None = None,
    future_feminine_stem: str | None = None,
    infinitive: str | None = None,
) -> VerbParadigm:
    """Third-person past, participle and future; no first/second persons.

    Weak roots supply their grammatical stems, rather than an exception list
    of inflected words. For final-ה verbs the feminine past takes תה, plural
    past/future drops ה, and feminine plural future takes ינה.
    """

    past_stem = past_stem or past
    if past.endswith("ה"):
        past_forms = (past, _joined(past[:-1], "תה"), _joined(past[:-1], "ו"))
    else:
        past_forms = (past, _joined(past_stem, "ה"), _joined(past_stem, "ו"))
    adjective = adjective_paradigm(present, feminine=feminine)
    if present.endswith("ה"):
        stem = present[:-1]
        present_forms = (
            present,
            _joined(present_feminine_stem, "ת")
            if present_feminine_stem
            else _joined(stem, "ה"),
            _joined(stem, "ים"),
            _joined(stem, "ות"),
        )
    else:
        present_forms = adjective.absolute
    feminine_future = future_feminine or "ת" + future[1:]
    future_plural = future_plural or (future[:-1] if future.endswith("ה") else future)
    if future_feminine_stem is not None:
        feminine_plural = _joined(future_feminine_stem, "נה")
    elif future.endswith("ה"):
        feminine_plural = _joined(feminine_future[:-1], "ינה")
    else:
        # A root-final נ merges with the נ of the feminine plural ending
        # (תמאנה), rather than spelling the same consonant twice.
        feminine_plural = _joined(
            feminine_future[:-1]
            if feminine_future.endswith(("ן", "נ"))
            else feminine_future,
            "נה",
        )
    return VerbParadigm(
        past,
        past_forms,
        present_forms,
        (future, feminine_future, _joined(future_plural, "ו"), feminine_plural),
        (infinitive,) if infinitive else (),
        {} if infinitive else {"infinitive": "finite predicate construction"},
    )


def compact_alternation(words: Iterable[str]) -> str:
    """A factored trie: shared stems and endings, rather than flat literals."""

    tree: dict = {}
    for word in sorted(set(words)):
        node = tree
        for char in word:
            node = node.setdefault(char, {})
        node[""] = {}

    def emit(node: dict) -> str:
        optional = "" in node
        alternatives = [
            re.escape(char) + emit(child) for char, child in node.items() if char
        ]
        if not alternatives:
            return ""
        joined = (
            alternatives[0]
            if len(alternatives) == 1
            else "(?:" + "|".join(alternatives) + ")"
        )
        return "(?:" + joined + ")?" if optional else joined

    return emit(tree)


# These already-listed prepositions contain a productive ב prefix. The
# remainder stays the same head; this introduces no lexical family.
_PREFIXED_SOURCE_HEADS = {
    "באפס": ("ב", "אפס"),
    "במקום": ("ב", "מקום"),
    "בשונה": ("ב", "שונה"),
}


def source_head_language(words: Iterable[str]) -> str:
    """Factor finite heads and put every listed productive prefix through joints."""

    words = tuple(words)
    infinitives = {
        form for family in VERB_PARADIGMS.values() for form in family.infinitive
    }
    plain = [
        word
        for word in words
        if word not in infinitives and word not in _PREFIXED_SOURCE_HEADS
    ]
    infinitive_stems = [word[1:] for word in words if word in infinitives]
    alternatives = [compact_alternation(plain)] if plain else []
    if infinitive_stems:
        alternatives.append(
            rf"{prefix_word('ל')}(?:{compact_alternation(infinitive_stems)})"
        )
    for prefix in {entry[0] for entry in _PREFIXED_SOURCE_HEADS.values()}:
        stems = [
            entry[1]
            for word in words
            if (entry := _PREFIXED_SOURCE_HEADS.get(word)) and entry[0] == prefix
        ]
        if stems:
            alternatives.append(
                rf"{prefix_word(prefix)}(?:{compact_alternation(stems)})"
            )
    return "|".join(alternatives)


NOUN_PARADIGMS = {
    singular: noun_paradigm(singular, plural)
    for singular, plural in (
        ("קביעה", "קביעות"),
        ("היקבעות", "היקבעויות"),
        ("חישוב", "חישובים"),
        ("עיבוד", "עיבודים"),
        ("החלה", "החלות"),
        ("יישום", "יישומים"),
        ("ישום", "ישומים"),
        ("תחולה", "תחולות"),
        ("הטלה", "הטלות"),
        ("השתה", "השתות"),
        ("חיוב", "חיובים"),
        ("היווצרות", "היווצרויות"),
        ("הווצרות", "הווצרויות"),
        ("התהוות", "התהוויות"),
        ("צמיחה", "צמיחות"),
        ("טענה", "טענות"),
        ("דיווח", "דיווחים"),
        ("הנחה", "הנחות"),
        ("אמירה", "אמירות"),
        ("גרסה", "גרסאות"),
        ("גירסה", "גירסאות"),
    )
}
ADJECTIVE_PARADIGMS = {
    name: adjective_paradigm(name, feminine=feminine)
    for name, feminine in (
        ("שלול", "he"),
        ("משולל", "tav"),
        ("נטול", "he"),
        ("נעדר", "tav"),
        ("חסר", "he"),
        ("מחוסר", "tav"),
        ("חסום", "he"),
        ("פטור", "he"),
        ("נפקד", "tav"),
        ("נכון", "he"),
        ("שקרי", "tav"),
        ("שגוי", "he"),
        ("חישובי", "tav"),
        ("מוחל", "tav"),
        ("שונה", "he"),
    )
}
VERB_PARADIGMS = {
    v.name: v
    for v in (
        verb_paradigm("נחסם", "נחסם", "ייחסם"),
        verb_paradigm("סוכל", "מסוכל", "יסוכל"),
        verb_paradigm("סיכל", "מסכל", "יסכל"),
        verb_paradigm("נמנע", "נמנע", "יימנע"),
        verb_paradigm("הוכחש", "מוכחש", "יוכחש"),
        verb_paradigm("מיאן", "ממאן", "ימאן"),
        verb_paradigm("התכחש", "מתכחש", "יתכחש"),
        verb_paradigm("התנער", "מתנער", "יתנער", infinitive="להתנער"),
        verb_paradigm("הוצא", "מוצא", "יוצא", infinitive="להוציא"),
        verb_paradigm("מנע", "מונע", "ימנע"),
        verb_paradigm("נדחה", "נדחה", "יידחה", present_feminine_stem="נדחי"),
        verb_paradigm("דחה", "דוחה", "ידחה"),
        verb_paradigm(
            "הבדיל",
            "מבדיל",
            "יבדיל",
            feminine="he",
            future_feminine_stem="תבדל",
            infinitive="להבדיל",
        ),
        verb_paradigm("כפר", "כופר", "יכפור", future_plural="יכפר"),
        verb_paradigm(
            "נותר", "נותר", "יוותר", future_feminine="תיוותר", infinitive="להיוותר"
        ),
        verb_paradigm("עמד", "עומד", "יעמוד", future_plural="יעמד", infinitive="לעמוד"),
        verb_paradigm("נשאר", "נשאר", "יישאר", infinitive="להישאר"),
        verb_paradigm(
            "המשיך",
            "ממשיך",
            "ימשיך",
            feminine="he",
            past_stem="המשיכ",
            future_plural="ימשיכ",
            future_feminine_stem="תמשכ",
        ),
        verb_paradigm("נבצר", "נבצר", "ייבצר"),
        verb_paradigm("נקבע", "נקבע", "ייקבע", infinitive="להיקבע"),
        verb_paradigm("חושב", "מחושב", "יחושב", infinitive="לחשב"),
        verb_paradigm("הוחל", "מוחל", "יוחל", infinitive="להחיל"),
        verb_paradigm("יושם", "מיושם", "ייושם", infinitive="ליישם"),
        verb_paradigm("הוטל", "מוטל", "יוטל", infinitive="להטיל"),
        verb_paradigm("חויב", "מחויב", "יחויב", infinitive="לחייב"),
        verb_paradigm("נוצר", "נוצר", "ייווצר", infinitive="להיווצר"),
        verb_paradigm("צמח", "צומח", "יצמח", infinitive="לצמוח"),
        verb_paradigm(
            "חל", "חל", "יחול", feminine="he", future_plural="יחול", infinitive="לחול"
        ),
        verb_paradigm("הושת", "מושת", "יושת", infinitive="להשית"),
        verb_paradigm(
            "קם",
            "קם",
            "יקום",
            feminine="he",
            future_plural="יקומ",
            future_feminine_stem="תקמ",
            infinitive="לקום",
        ),
    )
}


def _family_forms(names: Iterable[str], *, verbs: bool = True) -> tuple[str, ...]:
    registry = VERB_PARADIGMS if verbs else ADJECTIVE_PARADIGMS
    return _unique(form for name in names for form in registry[name].forms)


# A statutory homograph stays guarded; exclusions are recorded per form after
# counting the pinned 1,633-record snapshot under every ordinary prefix.
SOURCE_CLAUSE_NEGATORS = _unique(
    (
        *_family_forms(("נחסם", "סוכל", "סיכל", "נמנע", "הוכחש", "מיאן", "התכחש")),
        *_family_forms(("שלול", "משולל"), verbs=False),
    )
)
SOURCE_CLAUSE_PHRASES = (
    *(f"{form} מ" for form in VERB_PARADIGMS["התנער"].forms),
    *(f"{form} מכלל" for form in VERB_PARADIGMS["הוצא"].forms),
    *(
        f"{form} {complement}"
        for form in (
            "שקר",
            *ADJECTIVE_PARADIGMS["שקרי"].forms,
            *ADJECTIVE_PARADIGMS["שגוי"].forms,
        )
        for complement in ("ש", "כי")
    ),
)
SOURCE_PRIVATIVE_HEADS = _unique(
    (
        *_family_forms(("נטול", "נעדר", "חסר", "מחוסר", "משולל", "חסום"), verbs=False),
        *_family_forms(("מנע", "נדחה", "דחה")),
        "מניעת",
        "בלעדי",
        "באפס",
        "אי",
        "במקום",
        "חלף",
        "תחת",
    )
)
SOURCE_FROM_HEADS = _unique(
    (
        *ADJECTIVE_PARADIGMS["פטור"].absolute,
        *ADJECTIVE_PARADIGMS["שונה"].forms,
        *VERB_PARADIGMS["הבדיל"].forms,
        "בשונה",
        "להבדיל",
        "לבד",
    )
)
SOURCE_CONSTRUCT_HEADS = ADJECTIVE_PARADIGMS["פטור"].construct
SOURCE_ABSENT_FROM_PREDICATES = _family_forms(("נפקד", "נעדר", "חסר"), verbs=False)
SOURCE_QUANTIFIERS = ("כל", "שום", "אותו", "אותה", "אותם", "אותן")
ABSENCE_SUFFIXES = tuple(
    _POSSESSIVE_ENDINGS[0][slot] for slot in _THIRD_PERSON_POSSESSIVE_SLOTS
)
DENIAL_HEADS = VERB_PARADIGMS["כפר"].forms
_CLAIM_FAMILIES = ("טענה", "דיווח", "הנחה", "אמירה", "גרסה", "גירסה")
CLAIM_NOUNS = _unique(
    form for name in _CLAIM_FAMILIES for form in NOUN_PARADIGMS[name].forms
)
CLAIM_CONSTRUCT_NOUNS = _unique(
    form for name in _CLAIM_FAMILIES for form in NOUN_PARADIGMS[name].construct
)
CLAIM_NOUN_LANGUAGE = compact_alternation(CLAIM_NOUNS)
CLAIM_CONSTRUCT_LANGUAGE = compact_alternation(CLAIM_CONSTRUCT_NOUNS)


def claim_content(*, after: str, separator: str) -> str:
    """A shared claim noun/content clause, with at most three owner tokens.

    Only construct nouns take owners. Atomic Unicode-letter tokens cannot cross
    punctuation, clause boundaries or backtrack through a long word.
    """

    # Intersect the caller's separator with horizontal spaces and joints.
    # A \s+ separator must not carry a claim owner across a clause break.
    horizontal = "[" + re.escape("".join(sorted(HORIZONTAL_SPACES))) + "]"
    gap = rf"(?:{horizontal}+|{horizontal}*{PREFIX_JOINT}{horizontal}*)"
    separator = rf"(?={separator}){gap}"
    connective = rf"(?:{prefix_word('ש')}(?=[{_H}])|כי{after})"
    owner = rf"(?>[^\W\d_]+){after}{separator}"
    return (
        rf"(?:{prefix_word('ה')})?(?:"
        rf"(?:{CLAIM_NOUN_LANGUAGE}){after}{separator}{connective}"
        rf"|(?:{CLAIM_CONSTRUCT_LANGUAGE}){after}{separator}(?:{owner}){{1,3}}{connective})"
    )


DISCLAIMER_VERBS = _unique(
    form
    for name in ("נותר", "עמד", "נשאר")
    for slot in (
        VERB_PARADIGMS[name].past,
        VERB_PARADIGMS[name].present,
        VERB_PARADIGMS[name].future,
    )
    for form in slot
)
DISCLAIMER_INFINITIVES = _unique(
    form for name in ("נותר", "עמד", "נשאר") for form in VERB_PARADIGMS[name].infinitive
)
DISCLAIMER_CONTINUATIVES = VERB_PARADIGMS["המשיך"].forms
# בעינו/בעינה/בעינם/בעינן are the four third-person agreement slots.
DISCLAIMER_STATE_FORMS = tuple(_joined("בעינ", suffix) for suffix in ABSENCE_SUFFIXES)
POSTPOSED_CORRECT_FORMS = ADJECTIVE_PARADIGMS["נכון"].forms
POSTPOSED_FALSE_FORMS = _unique(
    (
        "שקר",
        *_family_forms(("שקרי", "שגוי"), verbs=False),
        *VERB_PARADIGMS["נבצר"].forms,
    )
)
# Copulas before the privative adjective (``יהיה מחוסר חישוב``).
COPULAS = frozenset(
    "היה היתה הייתה היו יהיה תהיה יהיו תהיינה להיות הוא היא הם הן הינו הינה "
    "הינם הינן הנו הנה הנם הנן".split()
)

_COMPUTATION_FAMILIES = tuple(
    name for name in NOUN_PARADIGMS if name not in _CLAIM_FAMILIES and name != "עיבוד"
)
COMPUTATION_NOUN_PARADIGMS = tuple(
    NOUN_PARADIGMS[name] for name in _COMPUTATION_FAMILIES
)
COMPUTATION_NOUN_FORMS = _unique(
    form for name in _COMPUTATION_FAMILIES for form in NOUN_PARADIGMS[name].forms
)
_COMPUTATION_NOUNS = (
    compact_alternation(COMPUTATION_NOUN_FORMS)
    + rf"|(?:{compact_alternation(NOUN_PARADIGMS['עיבוד'].forms)})\s+"
    + rf"(?:{prefix_word('ה')})?(?:{compact_alternation(ADJECTIVE_PARADIGMS['חישובי'].forms)})"
)
_COMPUTATION_VERB_FAMILIES = (
    "נקבע",
    "חושב",
    "הוחל",
    "יושם",
    "הוטל",
    "חויב",
    "נוצר",
    "צמח",
    "חל",
    "הושת",
    "קם",
)
COMPUTATION_VERB_PARADIGMS = tuple(
    VERB_PARADIGMS[name] for name in _COMPUTATION_VERB_FAMILIES
)
COMPUTATION_INFINITIVES = _unique(
    form
    for name in _COMPUTATION_VERB_FAMILIES
    for form in VERB_PARADIGMS[name].infinitive
)
# שיקום/שיקומו are statutory rehabilitation, not relative computational
# clauses. The feminine relative slots are unambiguous and stay admitted.
COMPUTATION_RELATIVE_FUTURES = _unique(
    form
    for name in _COMPUTATION_VERB_FAMILIES
    for form in VERB_PARADIGMS[name].future
    if not (name == "קם" and form.startswith("י"))
)
_COMPUTATION_VERBS = (
    source_head_language(COMPUTATION_INFINITIVES)
    + rf"|{prefix_word('ל')}היות\s+(?:{compact_alternation(ADJECTIVE_PARADIGMS['מוחל'].absolute)})"
    + rf"|{prefix_word('ש')}(?:{compact_alternation(COMPUTATION_RELATIVE_FUTURES)})"
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
    tails = {
        "מ": rf"{prefix_word('מ')}(?=[{_H}])",
        "מכלל": rf"{prefix_word('מ')}כלל{after}",
        "ש": rf"{prefix_word('ש')}(?=[{_H}])",
        "כי": rf"כי{after}",
    }
    clause_phrases = "|".join(
        rf"(?:{source_head_language(phrase.split()[0] for phrase in SOURCE_CLAUSE_PHRASES if phrase.split()[1] == tail)}){after}{gap}{pattern}"
        for tail, pattern in tails.items()
    )
    privative = source_head_language(
        head for head in SOURCE_PRIVATIVE_HEADS if head != "מחוסר"
    )
    claim = (
        rf"{prefix_word('ב')}{claim_content(after=after, separator=gap)}"
        rf"|{prefix_word('ב')}כך{after}{gap}(?:{prefix_word('ש')}(?=[{_H}])|כי{after})"
    )
    source_only = (
        rf"{before}(?P<source_prefix>{_NEGATION_PREFIX})(?:"
        rf"(?:{compact_alternation(SOURCE_CLAUSE_NEGATORS)}){after}"
        rf"|{clause_phrases}"
        rf"|(?P<guarded>)(?:"
        rf"(?:{privative}){after}(?={gap}{computation()})"
        rf"|(?P<lack>מחוסר){after}(?={gap}{computation(rf'(?P<lack_article>{article})')})"
        rf"|(?:{compact_alternation(ADJECTIVE_PARADIGMS[name].absolute[2] for name in ('נעדר', 'חסר'))}){after}(?={gap}{prefix_word('ב')}{article}{noun})"
        rf"|(?:{source_head_language(SOURCE_FROM_HEADS)}){after}(?={gap}?(?:{from_complement}))"
        rf"|(?:{compact_alternation(SOURCE_CONSTRUCT_HEADS)}){after}(?={gap}{computation()})"
        rf"|{prefix_word('ב')}(?:היעדר|העדר|חוסר)(?:{'|'.join(ABSENCE_SUFFIXES)}){after}"
        rf"(?={gap}(?:{prefix_word('ה')}מוחלט{after}{gap})?של{after}{gap}{computation()})"
        rf"|(?:{compact_alternation(DENIAL_HEADS)}){after}(?={gap}(?:{claim}))"
        rf"|{article}(?:{_COMPUTATION_NOUNS}|{_ENGLISH_COMPUTATION}){after}{gap}"
        rf"(?:{compact_alternation(SOURCE_ABSENT_FROM_PREDICATES)}){after}"
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


# The explicit complement disambiguates "exclude from" even after a modal
# which normally makes להוציא mean "issue". Only the source reader widens
# this reading: its newly closed הוצא מכלל paradigm includes the infinitive.
_EXPLICIT_SOURCE_EXCLUSION = re.compile(
    rf"{_NEGATION_GAP}+{prefix_word('מ')}כלל(?![{_H}])"
)


def _explicit_source_exclusion(match: re.Match[str]) -> bool:
    infinitive = match.group("exceptional_infinitive")
    return bool(
        infinitive
        and _EXPLICIT_SOURCE_EXCLUSION.match(
            match.string, match.end("exceptional_infinitive")
        )
    )


class NegationPattern:
    """A shared regex with statutory homograph exclusions in every reader."""

    def __init__(self, language: str) -> None:
        self._regex = re.compile(language)
        self.pattern = self._regex.pattern
        self._source = "lack" in self._regex.groupindex

    def _kept(self, match: re.Match[str]) -> bool:
        genuine = _genuine_context(match) or (
            self._source and _explicit_source_exclusion(match)
        )
        return genuine and not (
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
