"""Hebrew legal text for source-unit completeness.

Israeli statutes mark subsections with Hebrew letters (``(א)``, ``(ב)``,
``(ה1)``), paragraphs with numbers (``(1)``, ``(2א)``) and subparagraphs with
letters again. The Israel citation scheme ratified for rulespec-il on
2026-09-06 transliterates a letter label by its gematria value, not its sound:
values 1-26 map to ``a``-``z`` and 27 upward continue in bijective base 26. So
``(ב)(2)`` is the path ``b/2``, ``(ט)`` is ``i``, ``(יא)`` is ``k``, ``(כז)`` is
``aa`` and ``(ה1)`` is ``e1``.

This module recognizes that outline, reads native citations of the current
section (``סעיף 14(ב)(2)``), and holds the Hebrew forms of the deferral
precision rule. Each Hebrew form mirrors an English acceptance route in
:mod:`axiom_encode.harness.source_completeness` and is no looser than it.
``test_hebrew_rule_agrees_with_the_english_rule`` runs both rules on the same
text in both languages. The Hebrew rule also compares instruments, which the
English prose route does not; its one wider reading is a citation qualified
by its law (``סעיף 345ב לחוק הביטוח הלאומי``) before a missing state, which
the English reader refuses only because it does not parse the qualifier.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

HEBREW_LETTER_CLASS = "א-ת"
_H = HEBREW_LETTER_CLASS
_GERESH_MARKS = "׳״'\""
_MAQAF = "־"
_NOT_HEBREW_BEFORE = rf"(?<![{_H}A-Za-z0-9_])"
_NOT_HEBREW_AFTER = rf"(?![{_H}])"

HEBREW_LETTER = re.compile(rf"[{_H}]")

_LETTER_VALUES = {
    "א": 1,
    "ב": 2,
    "ג": 3,
    "ד": 4,
    "ה": 5,
    "ו": 6,
    "ז": 7,
    "ח": 8,
    "ט": 9,
    "י": 10,
    "כ": 20,
    "ך": 20,
    "ל": 30,
    "מ": 40,
    "ם": 40,
    "נ": 50,
    "ן": 50,
    "ס": 60,
    "ע": 70,
    "פ": 80,
    "ף": 80,
    "צ": 90,
    "ץ": 90,
    "ק": 100,
    "ר": 200,
    "ש": 300,
    "ת": 400,
}
_UNIT_LETTERS = ("", "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט")
_TEN_LETTERS = ("", "י", "כ", "ל", "מ", "נ", "ס", "ע", "פ", "צ")
_HUNDRED_LETTERS = ("", "ק", "ר", "ש")
_FINAL_LETTERS = "ךםןףץ"
_FINAL_FORMS = str.maketrans(_FINAL_LETTERS, "כמנפצ")
# A structural letter label is a small numeral. The cap keeps a short
# parenthesized word such as ``(של)`` from reading as label 330.
_MAX_LETTER_LABEL_VALUE = 99


def hebrew_numeral_value(text: str) -> int | None:
    """Return the value of a well-formed Hebrew numeral, or ``None``.

    Well-formed means the form :func:`hebrew_numeral` writes: letters in
    descending value, at most one of each of the hundreds (after any ``ת``),
    tens and units letters, ``טו``/``טז`` for 15 and 16, and no final forms,
    which numerals do not use.
    """

    letters = "".join(char for char in text if char not in _GERESH_MARKS)
    if not letters or any(
        char not in _LETTER_VALUES or char in _FINAL_LETTERS for char in letters
    ):
        return None
    tail = 0
    # 15 and 16 are written טו and טז, never יה and יו, also after hundreds.
    if letters.endswith(("טו", "טז")):
        tail = 9 + _LETTER_VALUES[letters[-1]]
        letters = letters[:-2]
        if any(_LETTER_VALUES[char] < 100 for char in letters):
            return None
    values = [_LETTER_VALUES[char] for char in letters]
    if values != sorted(values, reverse=True):
        return None
    hundreds = [value for value in values if 100 <= value < 400]
    tens = [value for value in values if 10 <= value < 100]
    units = [value for value in values if value < 10]
    if len(hundreds) > 1 or len(tens) > 1 or len(units) > 1:
        return None
    if tens == [10] and units and units[0] in {5, 6}:
        return None
    return sum(values) + tail


def hebrew_numeral(value: int) -> str:
    """Render a positive integer as a Hebrew numeral without geresh marks."""

    if value <= 0:
        raise ValueError(f"Hebrew numerals are positive: {value}")
    letters = "ת" * (value // 400)
    letters += _HUNDRED_LETTERS[value % 400 // 100]
    remainder = value % 100
    if remainder in {15, 16}:
        return letters + "ט" + _UNIT_LETTERS[remainder - 9]
    return letters + _TEN_LETTERS[remainder // 10] + _UNIT_LETTERS[remainder % 10]


def ordinal_path_label(value: int) -> str:
    """Map a gematria value to its path letters (1 -> a, 26 -> z, 27 -> aa)."""

    if value <= 0:
        raise ValueError(f"Ordinal path labels are positive: {value}")
    letters = []
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(reversed(letters))


def ordinal_path_value(label: str) -> int | None:
    """Invert :func:`ordinal_path_label`; ``None`` for anything but a-z letters."""

    if not re.fullmatch(r"[a-z]+", label):
        return None
    value = 0
    for letter in label:
        value = value * 26 + ord(letter) - ord("a") + 1
    return value


OUTLINE_LABEL = rf"(?:[{_H}]{{1,3}}\d{{0,2}}|\d{{1,3}}[{_H}]{{0,2}})"


@dataclass(frozen=True)
class HebrewOutlineLabel:
    """One parenthesized Hebrew outline label and its path transliteration."""

    marker: str
    kind: str
    value: int
    suffix: int

    @property
    def path_label(self) -> str:
        if self.kind == "letter":
            return ordinal_path_label(self.value) + (
                str(self.suffix) if self.suffix else ""
            )
        return str(self.value) + (
            ordinal_path_label(self.suffix) if self.suffix else ""
        )

    def follows(self, previous: HebrewOutlineLabel) -> bool:
        """Return whether this label is the next sibling after ``previous``.

        ``(ה)`` is followed by ``(ה1)`` or ``(ו)``, ``(ה1)`` by ``(ה2)`` or
        ``(ו)``; numbers likewise run ``(2)``, ``(2א)``, ``(3)``.
        """

        if self.kind != previous.kind:
            return False
        if self.value == previous.value and self.suffix == previous.suffix + 1:
            return True
        return self.value == previous.value + 1 and self.suffix == 0

    @property
    def is_first(self) -> bool:
        return self.value == 1 and self.suffix == 0


def parse_outline_label(text: str) -> HebrewOutlineLabel | None:
    """Parse ``ב``, ``ה1``, ``2`` or ``2א``; reject words and malformed numerals."""

    letters = re.fullmatch(rf"([{_H}]{{1,3}})(\d{{0,2}})", text)
    if letters is not None:
        value = hebrew_numeral_value(letters.group(1))
        digits = letters.group(2)
        if value is None or value > _MAX_LETTER_LABEL_VALUE or digits.startswith("0"):
            return None
        return HebrewOutlineLabel(text, "letter", value, int(digits or 0))
    number = re.fullmatch(rf"(\d{{1,3}})([{_H}]{{0,2}})", text)
    if number is None or number.group(1).startswith("0"):
        return None
    suffix = 0
    if number.group(2):
        suffix_value = hebrew_numeral_value(number.group(2))
        if suffix_value is None:
            return None
        suffix = suffix_value
    return HebrewOutlineLabel(text, "paragraph", int(number.group(1)), suffix)


_OUTLINE_CHAIN = re.compile(
    rf"(?m)^[ \t]*(?P<chain>\({OUTLINE_LABEL}\)(?:[ \t]*\({OUTLINE_LABEL}\))*)"
    r"(?=\s|$)"
)
_CHAIN_LABEL = re.compile(rf"\(({OUTLINE_LABEL})\)")


def _place_outline_label(
    active: list[HebrewOutlineLabel],
    label: HebrewOutlineLabel,
    *,
    opens_child: bool,
) -> list[HebrewOutlineLabel]:
    """Place one line-leading label in the open outline stack.

    A label continues the deepest open sequence it follows. When a shallower
    sequence also continues with it (subsection ``(ד)`` whose last
    subparagraph is ``(ד)``, then ``(ה)``), the text alone is ambiguous; a
    line that opens its own first child, as in ``(ה) (1)``, is read as the
    shallower level, which is how subsections open in Israeli statutes.
    """

    followed = [
        depth
        for depth in range(len(active) - 1, -1, -1)
        if label.follows(active[depth])
    ]
    if followed:
        depth = followed[-1] if opens_child else followed[0]
        return [*active[:depth], label]
    if active and label.is_first and active[-1].kind != label.kind:
        return [*active, label]
    for depth in range(len(active) - 1, -1, -1):
        if active[depth].kind == label.kind:
            return [*active[:depth], label]
    return [*active, label]


def hebrew_outline_markers(
    source_text: str,
) -> tuple[tuple[int, tuple[str, ...], str, str], ...]:
    """Return ``(start, path, kind, marker)`` for each line-leading outline label.

    Only a source with at least one line-leading Hebrew label uses this
    outline; anything else keeps the existing recognizers. A marker chain
    such as ``(ב) (1)`` nests each later label under the earlier one and
    shares its start, as the U.S. outline recognizer does for ``(a)(1)``.
    Inline references such as ``סעיף קטן (ב)(1)`` never start a line and are
    not markers.
    """

    if not HEBREW_LETTER.search(source_text):
        # Keep non-Hebrew sources at one C-speed scan: a CFR-style outline of
        # 20,000 `(n)` lines would otherwise parse every label here.
        return ()
    chains = []
    for match in _OUTLINE_CHAIN.finditer(source_text):
        labels = [
            parse_outline_label(text)
            for text in _CHAIN_LABEL.findall(match.group("chain"))
        ]
        if labels and all(label is not None for label in labels):
            chains.append((match.start("chain"), labels))
    if not any(
        HEBREW_LETTER.search(label.marker)
        for _start, labels in chains
        for label in labels
    ):
        return ()
    markers: list[tuple[int, tuple[str, ...], str, str]] = []
    active: list[HebrewOutlineLabel] = []
    for start, labels in chains:
        for index, label in enumerate(labels):
            active = (
                _place_outline_label(active, label, opens_child=len(labels) > 1)
                if index == 0
                else [*active, label]
            )
            markers.append(
                (
                    start,
                    tuple(item.path_label for item in active),
                    label.kind,
                    f"({label.marker})",
                )
            )
    return tuple(markers)


LEADING_OUTLINE_CHAIN = re.compile(rf"^\s*(?P<chain>(?:\({OUTLINE_LABEL}\)[ \t]*)+)")


def strip_leading_outline_chain(text: str) -> str | None:
    """Strip a leading Hebrew marker chain such as ``(ב) (1)``; else ``None``."""

    match = LEADING_OUTLINE_CHAIN.match(text)
    if match is None or not HEBREW_LETTER.search(match.group("chain")):
        return None
    labels = [
        parse_outline_label(label)
        for label in _CHAIN_LABEL.findall(match.group("chain"))
    ]
    if not labels or any(label is None for label in labels):
        return None
    return text[match.end() :].lstrip()


# ``(בוטל)`` is the consolidation's repeal note, the counterpart of the German
# ``(aufgehoben)`` that the editorial-omission recognizer already skips.
EDITORIAL_OMISSION_WORDS = r"בוטל(?:ה|ו)?|נמחק(?:ה|ו)?"


def is_hebrew_corpus_path(corpus_citation_path: str) -> bool:
    return corpus_citation_path.strip("/").lower().startswith("il/")


def native_section_label(corpus_citation_path: str) -> str | None:
    """Render an il ``section-<n><suffix>`` leaf as printed: ``section-121b`` -> ``121ב``."""

    if not is_hebrew_corpus_path(corpus_citation_path):
        return None
    leaf = corpus_citation_path.rstrip("/").rsplit("/", 1)[-1].lower()
    match = re.fullmatch(r"section-(\d+)([a-z]*)", leaf)
    if match is None:
        return None
    number, suffix = match.groups()
    if not suffix:
        return number
    value = ordinal_path_value(suffix)
    return number + hebrew_numeral(value) if value else None


def native_path_label(component: str) -> str | None:
    """Render one path component as its printed Hebrew label: ``e1`` -> ``ה1``."""

    number = re.fullmatch(r"(\d+)([a-z]*)", component)
    if number is not None:
        if not number.group(2):
            return number.group(1)
        value = ordinal_path_value(number.group(2))
        return number.group(1) + hebrew_numeral(value) if value else None
    letters = re.fullmatch(r"([a-z]+)(\d*)", component)
    if letters is None:
        return None
    value = ordinal_path_value(letters.group(1))
    if value is None or value > _MAX_LETTER_LABEL_VALUE:
        return None
    return hebrew_numeral(value) + letters.group(2)


def native_branch_citation(
    corpus_citation_path: str, path: tuple[str, ...]
) -> str | None:
    """Return ``סעיף 14(ב)(2)`` for ``il/.../section-14`` and ``("b", "2")``."""

    section = native_section_label(corpus_citation_path)
    labels = [native_path_label(component) for component in path]
    if section is None or any(label is None for label in labels):
        return None
    return f"סעיף {section}" + "".join(f"({label})" for label in labels)


def _native_label_pattern(label: str) -> str:
    """Allow geresh or gershayim between the letters of a printed label."""

    return f"[{_GERESH_MARKS}]?".join(re.escape(char) for char in label)


# A citation followed by another instrument (``לחוק הביטוח הלאומי``) names that
# instrument's section, not the current one; ``לחוק זה`` keeps the current law.
_OTHER_INSTRUMENT_QUALIFIER = (
    r"(?!\s*,?\s*(?:של\s+|ל|ב)ה?(?:חוק|פקוד[הת]|תקנות|צו)"
    rf"{_NOT_HEBREW_AFTER}"
    rf"(?!\s+(?:זה|זו|הזה|הזאת){_NOT_HEBREW_AFTER}))"
)
_PROCLITICS = "[ובלמשכה]{0,3}"
_SECTION_WORD = r"(?:סעיף|ס[\u05f3'])"


@functools.lru_cache(maxsize=4096)
def current_branch_citation_pattern(
    corpus_citation_path: str,
    path: tuple[str, ...],
    *,
    citation_must_start: bool = False,
    citation_must_end: bool = False,
) -> re.Pattern[str] | None:
    """Match a native citation of exactly this branch of the current section.

    The branch must be complete and terminal, as for a canonical citation:
    ``סעיף 14(ב)`` does not cite ``(ב)(2)`` and ``סעיף 14(ב)(2)(א)`` does not
    cite ``(ב)(2)``.
    """

    if not path:
        return None
    section = native_section_label(corpus_citation_path)
    labels = [native_path_label(component) for component in path]
    if section is None or any(label is None for label in labels):
        return None
    branch = r"\s*".join(
        rf"\(\s*{_native_label_pattern(label)}\s*\)" for label in labels
    )
    start_guard = r"^\s*" if citation_must_start else ""
    end_guard = r"\s*$" if citation_must_end else ""
    return re.compile(
        rf"{start_guard}{_NOT_HEBREW_BEFORE}{_PROCLITICS}{_SECTION_WORD}\s*"
        rf"{re.escape(section)}\s*{branch}"
        rf"(?!\s*\()(?![A-Za-z0-9_{_H}]){_OTHER_INSTRUMENT_QUALIFIER}{end_guard}"
    )


def native_source_reference_paths(
    source: str, corpus_citation_path: str
) -> set[tuple[str, ...]]:
    """Return branch paths a rule ``source:`` claims through native citations."""

    section = native_section_label(corpus_citation_path)
    if section is None:
        return set()
    paths: set[tuple[str, ...]] = set()
    for match in re.finditer(
        rf"{_NOT_HEBREW_BEFORE}{_PROCLITICS}{_SECTION_WORD}\s*{re.escape(section)}"
        rf"(?P<chain>(?:\s*\(\s*{OUTLINE_LABEL}\s*\))+)"
        rf"(?![A-Za-z0-9_{_H}]){_OTHER_INSTRUMENT_QUALIFIER}",
        source,
    ):
        labels = [
            parse_outline_label(label)
            for label in re.findall(
                rf"\(\s*({OUTLINE_LABEL})\s*\)", match.group("chain")
            )
        ]
        if labels and all(label is not None for label in labels):
            paths.add(tuple(label.path_label for label in labels))
    return paths


def cites_current_section(source: str, corpus_citation_path: str) -> bool:
    """Return whether text cites the current section natively (``סעיף 14``)."""

    section = native_section_label(corpus_citation_path)
    return bool(
        section
        and re.search(
            rf"{_NOT_HEBREW_BEFORE}{_PROCLITICS}{_SECTION_WORD}\s*{re.escape(section)}"
            rf"(?![0-9{_H}]){_OTHER_INSTRUMENT_QUALIFIER}",
            source,
        )
    )


# --- Deferral precision -----------------------------------------------------

_SIGNAL_PROCLITICS = r"(?:וש|שה|וה|ו|ש|ה)?"
_MISSING_STATE = (
    r"(?:חסר(?:ה|ים|ות)?"
    r"|טרם\s+קודד(?:ה|ו)?"
    r"|(?:אינו|אינה|אינם|אינן|לא)\s+(?:מקודד(?:ת|ים|ות)?|קודד(?:ה|ו)?"
    r"|זמין|זמינ(?:ה|ים|ות)))"
)
_UNTIL_STATE = (
    r"(?:יקודד(?:ו)?|תקודד(?:נה)?|ייקבע(?:ו)?|תיקבע"
    r"|יהיה\s+זמין|תהיה\s+זמינה|יהיו\s+זמינ(?:ים|ות))"
)

# The Hebrew counterparts of ``_MISSING_DEPENDENCY_LANGUAGE``: missing,
# requires/depends on, until, without, not (yet) encoded, unavailable and
# cannot be computed. A one-letter conjunction, relative or article prefix
# may attach (``וחסרים``, ``שחסרה``); the next letter must end the word.
MISSING_DEPENDENCY_LANGUAGE = (
    rf"{_NOT_HEBREW_BEFORE}{_SIGNAL_PROCLITICS}(?:"
    rf"{_MISSING_STATE}{_NOT_HEBREW_AFTER}"
    rf"|דורש(?:ת|ים|ות)?{_NOT_HEBREW_AFTER}"
    rf"|תלוי(?:ה|ים|ות)?\s+ב(?=[{_H}])"
    rf"|עד\s+(?:ש(?=[{_H}])|אשר\s)"
    rf"|(?:בלי|ללא|מבלי){_NOT_HEBREW_AFTER}"
    rf"|(?:לא|אי)\s+(?:ניתן|אפשר)\s+(?:לחשב|לקודד|להכריע|לקבוע){_NOT_HEBREW_AFTER}"
    r")"
)
_SIGNAL = re.compile(
    rf"{_NOT_HEBREW_BEFORE}{_SIGNAL_PROCLITICS}(?:"
    rf"(?P<missing>{_MISSING_STATE}){_NOT_HEBREW_AFTER}"
    rf"|(?P<requires>דורש(?:ת|ים|ות)?{_NOT_HEBREW_AFTER}|תלוי(?:ה|ים|ות)?\s+ב(?=[{_H}]|$))"
    rf"|(?P<until>עד\s+(?:ש(?P<until_verb>{_UNTIL_STATE}){_NOT_HEBREW_AFTER}"
    rf"|ש(?=[{_H}]|$)|אשר\s))"
    rf"|(?P<without>(?:בלי|ללא|מבלי){_NOT_HEBREW_AFTER})"
    rf"|(?P<cannot>(?:לא|אי)\s+(?:ניתן|אפשר)\s+(?:לחשב|לקודד|להכריע|לקבוע)"
    rf"{_NOT_HEBREW_AFTER})"
    r")"
)
_NEGATED_SIGNAL_PREFIX = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?(?:לא|אין|אינו|אינה|אינם|אינן|כלל\s+לא)\s*$"
)

# Counterparts of ``_ADVERSATIVE_LANGUAGE`` (but, however, although, despite,
# notwithstanding, whereas).
ADVERSATIVE_LANGUAGE = (
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?(?:אך|אבל|אולם|ברם|למרות|חרף){_NOT_HEBREW_AFTER}"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?(?:אף\s+על\s+פי|אף|אלא|בעוד)\s+ש(?=[{_H}])"
    rf"|{_NOT_HEBREW_BEFORE}(?:עם\s+זאת|על\s+אף){_NOT_HEBREW_AFTER}"
)
_ADVERSATIVE = re.compile(ADVERSATIVE_LANGUAGE)

_INSTRUMENT_KIND = r"(?:חוק|פקוד[הת]|תקנות|צו)"
SECTION_DEPENDENCY = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?P<proclitics>[ובלמשכה]{{0,3}})(?:סעיפים|סעיף)\s+"
    rf"(?P<section>\d+[{_H}]{{0,2}})(?![0-9{_H}])"
    rf"(?P<fragments>(?:\s*\(\s*{OUTLINE_LABEL}\s*\))*)"
    rf"(?P<instrument>\s*,?\s*(?:של\s+|ל|ב)ה?(?P<instrument_kind>{_INSTRUMENT_KIND})"
    rf"{_NOT_HEBREW_AFTER}"
    rf"(?P<instrument_name>(?:\s+[{_H}][{_H}{_GERESH_MARKS}{_MAQAF}-]*){{0,2}}))?"
)


def _normalize_word(text: str) -> str:
    return "".join(char for char in text if char not in _GERESH_MARKS).translate(
        _FINAL_FORMS
    )


# "This law", "these regulations": the current instrument.
_CURRENT_INSTRUMENT_NAMES = frozenset(
    map(_normalize_word, ("זה", "זו", "הזה", "הזאת", "אלה", "האלה"))
)
# "The said law" and its feminine and plural forms: the instrument last named.
_CONTEXTUAL_INSTRUMENT_NAMES = _CURRENT_INSTRUMENT_NAMES | frozenset(
    map(_normalize_word, ("האמור", "האמורה", "האמורים", "האמורות", "הנ״ל"))
)


def _dependency_identity(
    match: re.Match[str],
) -> tuple[str, tuple[str, ...]]:
    fragments = []
    for text in re.findall(rf"\(\s*({OUTLINE_LABEL})\s*\)", match.group("fragments")):
        label = parse_outline_label(text)
        fragments.append(label.path_label if label is not None else text)
    return _normalize_word(match.group("section")), tuple(fragments)


# A law, ordinance, regulations or order named anywhere in the text, so that
# ``לחוק האמור`` ("the said law") can be read as the instrument last named, as
# in ``קיצבאות כמשמעותן בחוק הביטוח הלאומי, לפי סעיף 2 לחוק האמור``.
_INSTRUMENT_MENTION = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?:של\s+|[ולבמשכ]{{0,2}})ה?(?P<kind>{_INSTRUMENT_KIND})"
    rf"{_NOT_HEBREW_AFTER}"
    rf"(?P<name>(?:\s+[{_H}][{_H}{_GERESH_MARKS}{_MAQAF}-]*){{0,2}})"
)
_CURRENT_INSTRUMENT = ("current", ())
_UNRESOLVED_INSTRUMENT = ("unresolved", ())
Instrument = tuple[str, tuple[str, ...]]


def _instrument_kind(kind: str) -> str:
    return "פקודה" if kind.startswith("פקוד") else kind


def _instrument_words(name: str) -> tuple[str, ...]:
    return tuple(_normalize_word(word) for word in name.split())


def _named_instrument(
    kind: str,
    words: tuple[str, ...],
    mentions: list[tuple[int, Instrument]],
    start: int,
) -> Instrument:
    """Resolve ``לחוק זה``, ``לחוק האמור`` or ``לחוק הביטוח הלאומי``."""

    first = words[0] if words else ""
    if first in _CURRENT_INSTRUMENT_NAMES:
        return _CURRENT_INSTRUMENT
    if words and first not in _CONTEXTUAL_INSTRUMENT_NAMES:
        return (kind, words)
    # A contextual (or bare) name is the same kind of instrument last named.
    return next(
        (
            instrument
            for position, instrument in reversed(mentions)
            if position < start and instrument[0] == kind
        ),
        _UNRESOLVED_INSTRUMENT,
    )


def _explicit_instrument_mentions(text: str) -> list[tuple[int, Instrument]]:
    mentions: list[tuple[int, Instrument]] = []
    for mention in _INSTRUMENT_MENTION.finditer(text):
        words = _instrument_words(mention.group("name"))
        if words and words[0] not in _CONTEXTUAL_INSTRUMENT_NAMES:
            mentions.append(
                (mention.start(), (_instrument_kind(mention.group("kind")), words))
            )
    return mentions


def _section_citations(
    text: str,
) -> list[tuple[re.Match[str], Instrument | None]]:
    """Return each section citation with the instrument it belongs to.

    ``None`` is a citation that names no instrument; in a source that is the
    current law, while a reason may leave the instrument to its context.
    """

    mentions = _explicit_instrument_mentions(text)
    citations: list[tuple[re.Match[str], Instrument | None]] = []
    for match in SECTION_DEPENDENCY.finditer(text):
        kind = match.group("instrument_kind")
        instrument = (
            _named_instrument(
                _instrument_kind(kind),
                _instrument_words(match.group("instrument_name") or ""),
                mentions,
                match.start("instrument"),
            )
            if kind
            else None
        )
        citations.append((match, instrument))
    return citations


def _same_instrument(left: Instrument, right: Instrument) -> bool:
    """Same kind, and one name is the other's leading words (``הביטוח``/``הביטוח הלאומי``)."""

    if _UNRESOLVED_INSTRUMENT in (left, right) or left[0] != right[0]:
        return False
    shorter, longer = sorted((left[1], right[1]), key=len)
    return longer[: len(shorter)] == shorter


def _dependency_is_external(
    match: re.Match[str], instrument: Instrument | None, current_section: str
) -> bool:
    if instrument is not None and instrument != _CURRENT_INSTRUMENT:
        return True
    section, _fragments = _dependency_identity(match)
    return section != _normalize_word(current_section)


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Bound a clause by ``;``, a line break or a sentence period, not ``5.17``."""

    stops = [match.start() for match in re.finditer(r"[;\n]|\.(?!\d)", text)]
    clause_start = max((stop for stop in stops if stop < start), default=-1) + 1
    clause_end = min((stop for stop in stops if stop >= end), default=len(text))
    return clause_start, clause_end


def _source_links_dependency(clause_before: str) -> bool:
    """Require operative syntax before a source citation of the dependency."""

    if re.search(
        r"(?:(?:מ)?בלי\s+לגרוע|על\s+אף|אין\s+ב[^.;\n]{0,80}כדי\s+לגרוע)"
        r"[^.;\n]{0,40}$",
        clause_before,
    ):
        return False
    return bool(
        re.search(
            r"(?:"
            rf"{_NOT_HEBREW_BEFORE}(?:ו|ש|וש)?(?:לפי|על[\s{_MAQAF}-]פי|עפ[״\"]י)\s*"
            rf"|{_NOT_HEBREW_BEFORE}(?:ו)?בהתאם\s+ל"
            rf"|{_NOT_HEBREW_BEFORE}(?:ו)?כאמור\s+ב"
            rf"|{_NOT_HEBREW_BEFORE}(?:ו)?כ(?:משמעות|הגדרת)(?:ו|ה|ם|ן)?\s+ב"
            rf"|{_NOT_HEBREW_BEFORE}(?:ו)?כ(?:מפורט|קבוע)\s+ב"
            rf"|{_NOT_HEBREW_BEFORE}(?:ו|ש)?(?:ה)?(?:קבוע|מנוי)(?:ה|ים|ות)?\s+ב"
            rf"|{_NOT_HEBREW_BEFORE}(?:ו)?(?:ה)?הוראות\s+(?:לפי\s*|ש?ב)?"
            r")$",
            clause_before,
        )
    )


def _source_scope_links_dependency(
    source_scope_text: str,
    dependency: re.Match[str],
    instrument: Instrument | None,
) -> bool:
    """Bind a reason's dependency to an operative citation in the deferred scope.

    The scope must cite the same section (and at least the reason's
    fragments) of the same instrument. A reason that names no instrument
    takes the one the scope cites; one that names an instrument binds only a
    scope citation of that instrument, with ``לחוק האמור`` read as the law
    last named and an unqualified scope citation read as the current law.
    """

    section, fragments = _dependency_identity(dependency)
    for cited, cited_instrument in _section_citations(source_scope_text):
        cited_section, cited_fragments = _dependency_identity(cited)
        if cited_section != section:
            continue
        if fragments and cited_fragments[: len(fragments)] != fragments:
            continue
        if instrument is not None and not _same_instrument(
            instrument, cited_instrument or _CURRENT_INSTRUMENT
        ):
            continue
        clause_start, _clause_end = _clause_bounds(
            source_scope_text, cited.start(), cited.end()
        )
        before = source_scope_text[clause_start : cited.start()] + cited.group(
            "proclitics"
        )
        if _source_links_dependency(before):
            return True
    return False


# Citations an English or canonical reader counts, so a mixed-language
# clause such as ``il/statute/.../section-14(e1): חסר ...`` is read as having
# two citations, as ``_PRECISE_DEFERRAL_DEPENDENCY`` reads it.
_OTHER_CITATION = re.compile(
    r"§|\b(?:section|subsection|paragraph)\s*\d"
    r"|\b[a-z]{2}(?:-[a-z0-9-]+)?(?:/(?:statute|regulation|guidance|manual)/"
    r"|:(?:statutes|regulations|guidance|manuals)/)",
    flags=re.IGNORECASE,
)


def _has_dependency(text: str) -> bool:
    return (
        SECTION_DEPENDENCY.search(text) is not None
        or _OTHER_CITATION.search(text) is not None
    )


# Counterparts of ``_DEPENDENCY_SUBJECT_TERMS``: the head noun of a bounded
# subject phrase such as ``התנאים`` or ``הוראות החישוב``. Hebrew puts the head
# first where English puts it last.
_DEPENDENCY_SUBJECT_WORDS = frozenset(
    map(
        _normalize_word,
        (
            "סכום סכומי סכומים סיוע גמלה גמלת גמלאות גימלה גימלת גימלאות קצבה "
            "קצבת קצבאות חישוב חישובי סיווג תנאי תנאים נתון נתונים נתוני מועד "
            "מועדי מועדים תאריך קביעה קביעת מסמך מסמכי מסמכים זכאות עובדה "
            "עובדות טופס טפסים הנחיה הנחיות הכנסה הכנסת מידע קלט קלטים תקרה "
            "תקרת הודעה הודעת תשלום תשלומי תשלומים שיעור שיעורי שיעורים רשומה "
            "רישום תקנה תקנות דרישה דרישות כלל כללי כללים הוראה הוראת הוראות "
            "תקן מעמד סף נוהל הליך תהליך הגשה אימות תכנית תוכנית"
        ).split(),
    )
)
# Counterparts of ``_DEPENDENCY_MODIFIER_TERMS`` (of, or, annual, applicable,
# administrative, gross, household, legal, local, public, state, tax, year).
_DEPENDENCY_MODIFIER_WORDS = _DEPENDENCY_SUBJECT_WORDS | frozenset(
    map(
        _normalize_word,
        (
            "של או וכן שנתי שנתית חודשי חודשית החל החלה החלים החלות מנהלי "
            "מנהלית ברוטו משק בית משפטי משפטית מקומי מקומית ציבורי ציבורית "
            "ממלכתי ממלכתית מס שנה שנת מתואם מתואמת ראשוני ראשונית"
        ).split(),
    )
)
_TERM_PREFIX_LETTERS = "ובהלמשכ"


def _is_term(word: str, terms: frozenset[str]) -> bool:
    """Whether ``word``, with up to two attached prefix letters, is in ``terms``."""

    candidate = _normalize_word(word)
    for _ in range(3):
        if candidate in terms:
            return True
        if len(candidate) < 3 or candidate[0] not in _TERM_PREFIX_LETTERS:
            return False
        candidate = candidate[1:]
    return False


def _subject_phrase_is_bounded(phrase: str) -> bool:
    """Counterpart of ``_dependency_subject_phrase_is_bounded``.

    A short noun phrase of listed words whose head is a subject word, such as
    ``התנאים`` or ``את הוראות החישוב``; anything else is open description.
    """

    if re.search(rf"[^{_H}{_GERESH_MARKS}{_MAQAF}\s,-]", phrase):
        return False
    words = re.findall(rf"[{_H}{_GERESH_MARKS}]+", phrase)
    if words and words[0] == "את":
        words = words[1:]
    return bool(
        words
        and _is_term(words[0], _DEPENDENCY_SUBJECT_WORDS)
        and all(_is_term(word, _DEPENDENCY_MODIFIER_WORDS) for word in words)
    )


# Counterparts of the English linkers ``under``, ``pursuant to``,
# ``according to`` and ``set``/``specified``/``provided``/``defined in``,
# ending where the dependency citation (with its prefix letters) begins.
_INTRODUCTION_LINKER = re.compile(
    r"(?:"
    rf"{_NOT_HEBREW_BEFORE}(?:ש)?(?:לפי|על[\s{_MAQAF}-]פי|עפ[״\"]י)\s*"
    rf"|{_NOT_HEBREW_BEFORE}בהתאם\s+ל"
    rf"|{_NOT_HEBREW_BEFORE}(?:ה|ש)?(?:קבוע|מנוי|מפורט|אמור|נדרש)(?:ה|ים|ות|ת)?\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}שנקבע(?:ו|ה)?\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}כ(?:אמור|מפורט|הגדרת(?:ו|ה|ם|ן)?|משמעות(?:ו|ה|ם|ן)?)\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}שב"
    r")$"
)


def _introduction_is_bounded(bridge: str) -> bool:
    """Counterpart of ``_reason_reference_introduction_is_bounded``.

    Between ``עד שיקודדו`` and the citation: nothing, or a bounded subject
    phrase and a linker, as in ``עד שיקודדו התנאים שבסעיף 345ב``.
    """

    if not bridge.strip():
        return True
    linker = _INTRODUCTION_LINKER.search(bridge)
    return linker is not None and _subject_phrase_is_bounded(bridge[: linker.start()])


_CANNOT_COMPUTE_BECAUSE = re.compile(
    r"^\s*(?:לא|אי)\s+(?:ניתן|אפשר)\s+(?:לחשב|לקודד|להכריע|לקבוע)"
    r"(?:\s+את\s+[^.;\n]{1,80}?)?\s+(?:כי|מאחר\s+ש|משום\s+ש|כיוון\s+ש)\s*"
)


def _requirement_subject_is_bounded(introduction: str) -> bool:
    """Counterpart of the English ``requires``/``depends on`` subject bound.

    As English strips ``cannot be computed because``, the Hebrew strips
    ``לא ניתן לחשב ... כי`` and requires the rest to be empty or a bounded
    subject phrase: ``החישוב תלוי ב...``, not ``הענף המיוחד הזה תלוי ב...``.
    """

    introduction = _CANNOT_COMPUTE_BECAUSE.sub("", introduction)
    return not introduction.strip() or _subject_phrase_is_bounded(introduction)


def _signals(text: str) -> list[re.Match[str]]:
    return [
        signal
        for signal in _SIGNAL.finditer(text)
        if not _NEGATED_SIGNAL_PREFIX.search(text[: signal.start()])
    ]


def _missing_state_follows(after: str) -> bool:
    """``…סעיף 345ב, שטרם קודד`` / ``…סעיף 345ב אינו מקודד`` and nothing more."""

    state = re.match(
        rf"\s*,?\s*(?:ש|אשר\s+)?{_MISSING_STATE}{_NOT_HEBREW_AFTER}", after
    )
    return state is not None and not after[state.end() :].strip(" \t,.)")


def _until_state_follows(after: str) -> bool:
    state = re.match(rf"\s*{_UNTIL_STATE}{_NOT_HEBREW_AFTER}", after)
    return state is not None and not after[state.end() :].strip(" \t,.)")


def reason_names_source_bound_dependency(
    reason: str,
    source_scope_text: str,
    *,
    corpus_citation_path: str,
) -> bool:
    """Accept a Hebrew reason naming a missing external section the scope applies.

    The Hebrew form of the English external-dependency route. The cited
    section must differ from the current one (or belong to another
    instrument), a missing-dependency signal must govern it within one
    clause with no adversative, negation or other citation in between, and
    the deferred source scope must make that section of that instrument
    operative (``לפי סעיף 345ב``, ``כהגדרתה בסעיף 345ב``). A reason that only
    describes its own branch, or names a section or instrument the scope does
    not apply, stays imprecise. This is stricter than the English prose route,
    which does not compare instruments.
    """

    current_section = native_section_label(corpus_citation_path)
    if current_section is None or not source_scope_text.strip():
        return False
    for dependency, instrument in _section_citations(reason):
        if not _dependency_is_external(dependency, instrument, current_section):
            continue
        clause_start, clause_end = _clause_bounds(
            reason, dependency.start(), dependency.end()
        )
        # A signal may be glued to the citation's prefix letters, as in
        # ``תלוי בסעיף`` (depends on section) or ``עד שסעיף`` (until section).
        before = reason[clause_start : dependency.start()] + dependency.group(
            "proclitics"
        )
        after = reason[dependency.end() : clause_end]
        if _ADVERSATIVE.search(after) or _ADVERSATIVE.search(before):
            continue
        signals = _signals(before)
        if not signals:
            continue
        signal = signals[-1]
        bridge = before[signal.end() :]
        if len(bridge) > 240 or _has_dependency(bridge):
            continue
        if signal.group("missing") or signal.group("without"):
            # As in English, an earlier citation in the same clause leaves the
            # missing claim ambiguous between the two.
            names_missing = not _has_dependency(before[: signal.start()])
        elif signal.group("requires"):
            names_missing = _requirement_subject_is_bounded(
                before[: signal.start()]
            ) and _missing_state_follows(after)
        elif signal.group("until"):
            names_missing = (
                signal.group("until_verb") is not None
                and _introduction_is_bounded(bridge)
                and not after.strip(" \t,.)")
            ) or (
                signal.group("until_verb") is None
                and not bridge.strip()
                and _until_state_follows(after)
            )
        else:
            # ``לא ניתן לחשב`` only introduces; the citation must carry its
            # own missing state (``כי סעיף 345ב טרם קודד``).
            names_missing = re.fullmatch(
                r"\s*(?:את\s+[^.;\n]{1,80}?\s+)?(?:כי|מאחר\s+ש|משום\s+ש|כיוון\s+ש)?\s*",
                bridge,
            ) is not None and _missing_state_follows(after)
        if names_missing and _source_scope_links_dependency(
            source_scope_text, dependency, instrument
        ):
            return True
    return False


# Counterparts of ``_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE``, word for word:
# administrative, calendar, capability, classification, complaint, content,
# data, date arithmetic, designation, determination, document, event, fact,
# filing, form, hearing, information, input, investigation, notice, process,
# procedure, record, relation, representation, status, submission, timing.
# Membership is left out (חברות also means "companies"), and relation keeps
# only יחסי, since ביחס ל is the preposition "with respect to".
RUNTIME_GAP_LANGUAGE = (
    rf"{_NOT_HEBREW_BEFORE}[ובהלמשכ]{{0,3}}(?:"
    r"מי?נהלי(?:ת|ים|ות)?|לוח\s+(?:שנה|זמנים)|יכולת|יכולות|סיווג(?:ים|י)?"
    r"|תלונ(?:ה|ות|ת)|תוכן|תכנים|נתונ(?:ים|י)|נתון|חישוב\s+תאריכים|ייעוד"
    r"|קביע(?:ה|ת|ות)|הכרע(?:ה|ת|ות)|מסמ(?:ך|כים|כי)|אירוע(?:ים|י)?|עובדה"
    r"|הגש(?:ה|ת|ות)|טופס|טפסים|טופסי|שימוע(?:ים)?|מידע|קלט(?:ים)?"
    r"|חקיר(?:ה|ת|ות)|הודע(?:ה|ות|ת)|תהליכ(?:ים|י)|תהליך|הליכ(?:ים|י)|הליך"
    r"|נוהל|נהלים|רשומ(?:ה|ות|ת)|רישום|רישומ(?:ים|י)|זיקה|יחסי|ייצוג"
    r"|מעמד|סטטוס|עיתוי|מועד(?:ים|י)?"
    rf"){_NOT_HEBREW_AFTER}"
)

# Counterparts of ``_ADMINISTRATIVE_SOURCE_ARTIFACT_LANGUAGE``: amendment,
# approval, audit, certification, comment, complaint, consultation, copy,
# document, filing, hearing, inspection, notice, plan, procedure,
# recommendation, record, report or return, submission, waiver.
ADMINISTRATIVE_ARTIFACT_LANGUAGE = (
    rf"{_NOT_HEBREW_BEFORE}[ובהלמשכ]{{0,3}}(?:"
    r"תיקונ(?:ים|י)|תיקון|אישור(?:ים|י|ו|ה)?|ביקורת|תעוד(?:ה|ות|ת)|הער(?:ה|ות|ת)"
    r"|תלונ(?:ה|ות|ת)|התייעצות|העתק(?:ים|י)?|מסמ(?:ך|כים|כי)|הגש(?:ה|ת|ות)"
    r"|שימוע(?:ים)?|פיקוח|הודע(?:ה|ות|ת)|תו?כני(?:ת|ות)|נוהל|נהלים|הליך|הליכים"
    r"|המלצ(?:ה|ות|ת)|רשומ(?:ה|ות|ת)|רישום|דוח(?:ות)?|דו[״\"]ח(?:ות)?"
    r"|דין\s+וחשבון|ויתור"
    rf"){_NOT_HEBREW_AFTER}"
)

# Counterparts of ``_ADMINISTRATIVE_SOURCE_ACTION_LANGUAGE``: adopt, approve,
# certify, consult, contain, conduct, enforce, establish, file, investigate,
# notify, prescribe, publish, recommend, review, submit, waive.
ADMINISTRATIVE_ACTION_LANGUAGE = (
    rf"{_NOT_HEBREW_BEFORE}(?:ו|ש|וש|כש)?(?:"
    r"לאמץ|יאמץ|תאמץ|אימץ|לאשר|יאשר|תאשר|אישר(?:ה|ו)?|מאשר(?:ת)?|יאושר(?:ו)?"
    r"|תאושר|אושר(?:ה|ו)?|להתייעץ|יתייעץ|תתייעץ|להכיל|יכיל|תכיל|מכיל(?:ה)?"
    r"|לערוך|יערוך|תערוך|לאכוף|יאכוף|תאכוף|להקים|יקים|תקים|הוקמ(?:ה|ו)|הוקם"
    r"|להגיש|יגיש(?:ו)?|תגיש|הגיש(?:ה|ו)?|יוגש(?:ו)?|תוגש|הוגש(?:ה|ו)?"
    r"|לחקור|יחקור|תחקור|להודיע|יודיע(?:ו)?|תודיע|הודיע(?:ה|ו)?"
    r"|לקבוע|יקבע(?:ו)?|תקבע|קבע(?:ה|ו)?|ייקבע(?:ו)?|תיקבע|נקבע(?:ה|ו)?"
    r"|לפרסם|יפרסם|תפרסם|פרסם|יפורסמו|יפורסם|תפורסם|פורסמ(?:ה|ו)|פורסם"
    r"|להמליץ|ימליץ|תמליץ|המליץ|לבחון|יבחן|תבחן|לעיין|יעיין|תעיין"
    r"|לוותר|יוותר|תוותר"
    rf"){_NOT_HEBREW_AFTER}"
)

_RUNTIME_GAP_STOPWORDS = frozenset(
    _normalize_word(word)
    for word in (
        "סעיף",
        "סעיפים",
        "קטן",
        "קטנים",
        "פסקה",
        "חסר",
        "חסרה",
        "חסרים",
        "חסרות",
        "קלט",
        "קלטים",
        "ייצוג",
        "יכולת",
        "מקודד",
        "קודד",
        "זמין",
        "ניתן",
        "לחשב",
        "לקודד",
        "לפי",
        "אשר",
        "הוראות",
        "הוראה",
        "חוק",
        "לחוק",
        "בחוק",
        "המקור",
        "מקור",
    )
)
_PREFIX_LETTERS = "ובהלמשכ"


def _word_variants(word: str) -> set[str]:
    """Return a word and its forms without up to two attached prefix letters."""

    normalized = _normalize_word(word)
    variants = {normalized}
    stripped = normalized
    for _ in range(2):
        if len(stripped) > 4 and stripped[0] in _PREFIX_LETTERS:
            stripped = stripped[1:]
            variants.add(stripped)
    return {variant for variant in variants if len(variant) >= 4}


def shared_substantive_word_count(reason: str, source_scope_text: str) -> int:
    """Count reason words of four or more letters that the source scope shares.

    Hebrew counterpart of the English rule that two shared substantive terms
    keep an exact self-citation from laundering a vague capability claim.
    """

    source_variants: set[str] = set()
    for word in re.findall(rf"[{_H}]+", source_scope_text):
        source_variants |= _word_variants(word)
    shared = set()
    for word in re.findall(rf"[{_H}]+", reason):
        variants = _word_variants(word)
        if not variants or variants & _RUNTIME_GAP_STOPWORDS:
            continue
        if variants & source_variants:
            shared.add(min(variants, key=len))
    return len(shared)


# A Hebrew branch states a computation when it applies a rate or fraction to
# a base (``בשיעור של 5.17% מהכנסתו``, ``מחצית``), names a product or
# difference (``מכפלה``, ``בניכוי``, ``ההפרש``) or directs a calculation
# (``יחושב``). ``בתוספת`` and ``חלקי`` are left out: in statutes they as often
# mean "together with" and "partial". Used only by the deferral guards.
_COMPUTATION_WINDOW = r"(?:[^.;\n]|(?<=\d)\.(?=\d))"
_COMPUTATION_BASE = r"(?:הכנס|שכר|סכום|גי?מל)"
_EXPLICIT_COMPUTATION = re.compile(
    rf"\d+(?:[.,]\d+)?\s*%\s*(?:ו)?מ(?:ן\s)?(?=[{_H}])"
    rf"|{_NOT_HEBREW_BEFORE}אחוז(?:ים)?\s+(?:ו)?מ(?:ן\s)?(?=[{_H}])"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?ב(?:ה)?שיעור(?:ים)?{_NOT_HEBREW_AFTER}"
    rf"{_COMPUTATION_WINDOW}{{0,120}}"
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?מ(?:ה)?{_COMPUTATION_BASE}"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?מ(?:ה)?{_COMPUTATION_BASE}[{_H}]*"
    rf"{_COMPUTATION_WINDOW}{{0,60}}{_NOT_HEBREW_BEFORE}(?:ו)?ב(?:ה)?שיעור(?:ים)?"
    rf"{_NOT_HEBREW_AFTER}"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?על\s+החלק\s+מ(?:ה)?{_COMPUTATION_BASE}"
    rf"|{_NOT_HEBREW_BEFORE}[ובהלש]{{0,2}}(?:מכפלה|מכפלת|מוכפל(?:ת|ים|ות)?|כפול"
    r"|מחולק(?:ת|ים|ות)?|בניכוי|ההפרש|הפרש|מחצית|שליש|רבע|פעמיים|כפליים"
    rf"|יחושב(?:ו)?|תחושב|יעוגל(?:ו)?){_NOT_HEBREW_AFTER}"
)


def states_explicit_computation(text: str) -> bool:
    return _EXPLICIT_COMPUTATION.search(text) is not None
