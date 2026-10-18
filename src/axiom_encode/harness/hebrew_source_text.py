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
text in both languages and holds that no Hebrew reason passes whose English
translation fails. Where they differ, the Hebrew is stricter: it also
compares instruments, which the English prose route does not.
"""

from __future__ import annotations

import bisect
import functools
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

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


# A label may carry a geresh or gershayim between or after its letters
# (``(י״א)``, ``(ה׳)``), as consolidations print numerals.
_LABEL_LETTERS = rf"[{_H}](?:[{_GERESH_MARKS}]?[{_H}]){{0,2}}[{_GERESH_MARKS}]?"
OUTLINE_LABEL = (
    rf"(?:{_LABEL_LETTERS}\d{{0,2}}"
    rf"|\d{{1,3}}(?:[{_H}](?:[{_GERESH_MARKS}]?[{_H}])?[{_GERESH_MARKS}]?)?)"
)
# A label with at least one Hebrew letter, for patterns shared with English
# text, where a bare ``(1)`` must keep its English reading.
HEBREW_OUTLINE_LABEL = rf"(?=[^()\s]*[{_H}]){OUTLINE_LABEL}"
# Israeli statutes nest subsection, paragraph, subparagraph and at most a few
# levels more. A deeper outline or citation chain is not one this reader
# supports: the outline is not used and the citation is not read, so neither
# can cost more than linear time.
MAX_OUTLINE_DEPTH = 6


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
    """Parse ``ב``, ``ה1``, ``י״א``, ``2`` or ``2א``; reject words and malformed numerals."""

    marker = text
    text = "".join(char for char in text if char not in _GERESH_MARKS)
    letters = re.fullmatch(rf"([{_H}]{{1,3}})(\d{{0,2}})", text)
    if letters is not None:
        value = hebrew_numeral_value(letters.group(1))
        digits = letters.group(2)
        if value is None or value > _MAX_LETTER_LABEL_VALUE or digits.startswith("0"):
            return None
        return HebrewOutlineLabel(marker, "letter", value, int(digits or 0))
    number = re.fullmatch(rf"(\d{{1,3}})([{_H}]{{0,2}})", text)
    if number is None or number.group(1).startswith("0"):
        return None
    suffix = 0
    if number.group(2):
        suffix_value = hebrew_numeral_value(number.group(2))
        if suffix_value is None:
            return None
        suffix = suffix_value
    return HebrewOutlineLabel(marker, "paragraph", int(number.group(1)), suffix)


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
            if len(active) > MAX_OUTLINE_DEPTH:
                return ()
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
    """Allow geresh or gershayim between or after the letters of a printed label."""

    marks = f"[{_GERESH_MARKS}]?"
    return marks.join(re.escape(char) for char in label) + marks


# A citation followed by another instrument (``לחוק הביטוח הלאומי``) names that
# instrument's section, not the current one; ``לחוק זה`` keeps the current law.
_INSTRUMENT_KIND = r"(?:חוק|פקוד[הת]|תקנות|צו)"
_QUOTES = "״\"׳'“”‘’«»„"
# How a citation names its instrument: ``לחוק``, ``בחוק``, ``מחוק``, ``מן
# החוק``, ``מתוך חוק``, ``של החוק``, each optionally relative (``שבחוק``,
# ``שמחוק``, ``אשר מתוך חוק``) and with the title optionally quoted
# (``ל״חוק הביטוח הלאומי״``).
_INSTRUMENT_INTRO = (
    r"(?:של\s+|ל|(?:ש|אשר\s+)?(?:ב|מ|מן\s+|מתוך\s+))"
    rf"[{_QUOTES}]?ה?[{_QUOTES}]?"
)
_OTHER_INSTRUMENT_QUALIFIER = (
    rf"(?!\s*(?:,\s*)?{_INSTRUMENT_INTRO}{_INSTRUMENT_KIND}"
    rf"{_NOT_HEBREW_AFTER}"
    rf"(?!\s+(?:זה|זו|הזה|הזאת){_NOT_HEBREW_AFTER}))"
)
_PROCLITICS = "[ובלמשכה]{0,3}"
# A section's letter suffix, with a geresh or gershayim as printed (``345י״א``).
_SECTION_SUFFIX = rf"(?:[{_H}](?:[{_GERESH_MARKS}]?[{_H}])?[{_GERESH_MARKS}]?)?"


def _native_section_pattern(section: str) -> str:
    """Match a printed section label, its suffix letters with optional marks."""

    number = re.match(r"\d+", section)
    digits = number.group(0) if number else ""
    letters = section[len(digits) :]
    return re.escape(digits) + (_native_label_pattern(letters) if letters else "")


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
        rf"{_native_section_pattern(section)}\s*{branch}"
        rf"(?!\s*\()(?![A-Za-z0-9_{_H}]){_OTHER_INSTRUMENT_QUALIFIER}{end_guard}"
    )


def _preposed_other_instrument(text: str, start: int) -> bool:
    """Whether another law is named just before ``start`` (``חוק הביטוח הלאומי, סעיף``)."""

    return any(
        match.end() == start
        and _instrument_words(match.group("name"))[:1]
        not in {(word,) for word in _CURRENT_INSTRUMENT_NAMES}
        for match in _PREPOSED_INSTRUMENT.finditer(text, max(0, start - 160), start)
    )


def cites_current_branch(
    text: str,
    corpus_citation_path: str,
    path: tuple[str, ...],
    *,
    citation_must_start: bool = False,
    citation_must_end: bool = False,
) -> bool:
    """Whether ``text`` cites exactly this branch of the current section natively."""

    pattern = current_branch_citation_pattern(
        corpus_citation_path,
        path,
        citation_must_start=citation_must_start,
        citation_must_end=citation_must_end,
    )
    return pattern is not None and any(
        not _preposed_other_instrument(text, match.start())
        for match in pattern.finditer(text)
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
        rf"{_NOT_HEBREW_BEFORE}{_PROCLITICS}{_SECTION_WORD}\s*{_native_section_pattern(section)}"
        rf"(?P<chain>(?:\s*\(\s*{OUTLINE_LABEL}\s*\)){{1,{MAX_OUTLINE_DEPTH}}})"
        rf"(?!\s*\()(?![A-Za-z0-9_{_H}]){_OTHER_INSTRUMENT_QUALIFIER}",
        source,
    ):
        if _preposed_other_instrument(source, match.start()):
            continue
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
    return bool(section) and any(
        not _preposed_other_instrument(source, match.start())
        for match in re.finditer(
            rf"{_NOT_HEBREW_BEFORE}{_PROCLITICS}{_SECTION_WORD}\s*{_native_section_pattern(section)}"
            rf"(?![0-9{_H}{_GERESH_MARKS}])(?:\s*\(\s*{OUTLINE_LABEL}\s*\)){{0,{MAX_OUTLINE_DEPTH}}}(?!\s*\()"
            rf"{_OTHER_INSTRUMENT_QUALIFIER}",
            source,
        )
    )


# --- Deferral precision -----------------------------------------------------

_SIGNAL_PROCLITICS = r"(?:וש|שה|וה|ו|ש|ה)?"
# A missing signal before its object: missing, not yet encoded, unavailable,
# the English signal words. Plain "not encoded" is not one in English either.
_MISSING_SIGNAL = (
    r"(?:חסר(?:ה|ים|ות)?"
    r"|(?:טרם|עדיין\s+(?:לא|אינו|אינה|אינם|אינן))\s+"
    r"(?:קודד(?:ה|ו)?|מקודד(?:ת|ים|ות)?)"
    r"|(?:אינו|אינה|אינם|אינן|לא)\s+(?:זמין|זמינ(?:ה|ים|ות)))"
)
# A state after its subject (``סעיף 345ב אינו מקודד``), where English also
# reads ``section 345b is not encoded``.
_MISSING_STATE = (
    rf"(?:{_MISSING_SIGNAL}"
    r"|(?:אינו|אינה|אינם|אינן|לא)\s+(?:מקודד(?:ת|ים|ות)?|קודד(?:ה|ו)?))"
)
_UNTIL_STATE = (
    r"(?:יקודד(?:ו)?|תקודד(?:נה)?|ייקבע(?:ו)?|תיקבע"
    r"|יהיה\s+זמין|תהיה\s+זמינה|יהיו\s+זמינ(?:ים|ות))"
)

# The Hebrew counterparts of ``_MISSING_DEPENDENCY_LANGUAGE``: missing,
# requires/depends on, until, without, not yet encoded, unavailable and
# cannot be computed. A one-letter conjunction, relative or article prefix
# may attach (``וחסרים``, ``שחסרה``); the next letter must end the word.
MISSING_DEPENDENCY_LANGUAGE = (
    rf"{_NOT_HEBREW_BEFORE}{_SIGNAL_PROCLITICS}(?:"
    rf"{_MISSING_SIGNAL}{_NOT_HEBREW_AFTER}"
    rf"|דורש(?:ת|ים|ות)?{_NOT_HEBREW_AFTER}"
    rf"|תלוי(?:ה|ים|ות)?\s+ב(?=[{_H}])"
    rf"|עד\s+(?:ש(?=[{_H}])|אשר\s)"
    rf"|(?:בלי|ללא|מבלי){_NOT_HEBREW_AFTER}"
    rf"|(?:לא|אי)\s+(?:ניתן|אפשר)\s+(?:לחשב|לקודד){_NOT_HEBREW_AFTER}"
    r")"
)
_SIGNAL = re.compile(
    rf"{_NOT_HEBREW_BEFORE}{_SIGNAL_PROCLITICS}(?:"
    rf"(?P<missing>{_MISSING_SIGNAL}){_NOT_HEBREW_AFTER}"
    rf"|(?P<requires>דורש(?:ת|ים|ות)?{_NOT_HEBREW_AFTER}|תלוי(?:ה|ים|ות)?\s+ב(?=[{_H}]|$))"
    rf"|(?P<until>עד\s+(?:(?:ש|אשר\s+)(?P<until_verb>{_UNTIL_STATE}){_NOT_HEBREW_AFTER}"
    rf"|ש(?=[{_H}]|$)|אשר\s))"
    rf"|(?P<without>(?:בלי|ללא|מבלי){_NOT_HEBREW_AFTER})"
    rf"|(?P<cannot>(?:לא|אי)\s+(?:ניתן|אפשר)\s+(?:לחשב|לקודד)"
    rf"{_NOT_HEBREW_AFTER})"
    r")"
)
_NEGATED_SIGNAL_PREFIX = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?:ו|ש|וש|כש)?(?:לא|אין|אינו|אינה|אינם|אינן|כלל\s+לא)\s*$"
)

# Words of a compound connective are separated by a space, a maqaf or a hyphen.
_SEP = rf"(?:\s+|\s*[{_MAQAF}-]\s*)"


def _connective(words: str) -> str:
    return _SEP.join(re.escape(word) for word in words.split())


# Counterparts of ``_ADVERSATIVE_LANGUAGE``: although, but, despite, even
# though, except, however, nevertheless, nonetheless, notwithstanding,
# though, unless, whereas, while, yet. Any may take the conjunction ``ו``.
_ADVERSATIVE_WORDS = (
    'אך אבל אולם ברם למרות חרף למעט זולת מלבד להוציא ואילו בעוד מאידך אעפ״כ אעפ"כ'
)
_ADVERSATIVE_PHRASES = (
    "אף כי",
    "אם כי",
    "אף על פי",
    "בכל זאת",
    "יחד עם זאת",
    "עם זאת",
    "על אף",
    "אלא אם",
    "גם אם",
)
# Connectives glued to the next word's first letter (``פרט לסעיף``, ``אף
# שסעיף``), which may also follow a separator (``פרט־ל``).
_ADVERSATIVE_GLUED = (
    ("פרט", "ל"),
    ("חוץ", "מ"),
    ("לבד", "מ"),
    ("הגם", "ש"),
    ("אף", "ש"),
    ("אלא", "ש"),
    ("גם אם", "ש"),
    ("אף על פי", "ש"),
    ("על אף העובדה", "ש"),
)
ADVERSATIVE_LANGUAGE = (
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?(?:"
    + "|".join(re.escape(word) for word in _ADVERSATIVE_WORDS.split())
    + "|"
    + "|".join(_connective(phrase) for phrase in _ADVERSATIVE_PHRASES)
    + rf"){_NOT_HEBREW_AFTER}"
    + rf"|{_NOT_HEBREW_BEFORE}(?:ו)?(?:"
    + "|".join(
        rf"{_connective(phrase)}(?:{_SEP})?{glue}"
        for phrase, glue in _ADVERSATIVE_GLUED
    )
    + rf")(?=[{_H}])"
)
_ADVERSATIVE = re.compile(ADVERSATIVE_LANGUAGE)

# Words that end a law's name: what follows a name in a statute or a reason
# (``בשינויים המחויבים``, ``טרם קודד``, ``לפי``), never a word of it.
_NAME_STOP_WORDS = (
    "בשינויים המחויבים לעניין לענין לפי כאמור כפי יחול יחולו ויחול ויחולו "
    "טרם אינו אינה אינם אינן לא חסר חסרה חסרים חסרות שטרם שאינו שלא ולכן לכן "
    "אשר כי וכן או אם גם רק אלא אך אבל אולם ואולם ובלבד עד יקודד קודד מקודד "
    "זמין ישולם ישלם חייב חייבת רשאי מחייב מחייבת קובע קובעת"
)
# Up to six name words and a parenthetical that distinguishes laws of one
# name (``חוק שירות המדינה (גמלאות)``, ``חוק הפיקוח על שירותים פיננסיים
# (ביטוח)``).
_NAME_WORD = (
    r"\s+(?!(?:"
    + "|".join(_NAME_STOP_WORDS.split())
    + rf"){_NOT_HEBREW_AFTER})[{_QUOTES}]?[{_H}][{_H}{_GERESH_MARKS}{_MAQAF}{_QUOTES}-]*"
)
_INSTRUMENT_NAME = (
    # ``חוק יסוד: כבוד האדם וחירותו`` and ``חוק־יסוד: חופש העיסוק``.
    rf"(?:\s*[{_MAQAF}-]?\s*יסוד\s*:)?"
    rf"(?:{_NAME_WORD}){{0,6}}"
    r"(?:\s*\([^()\n]{1,40}\))?"
)
# What may follow a name that was cut short: another name word past the
# six-word bound, a parenthetical too long or nested to read, or a quoted or
# colon-led continuation. Such a name identifies no law with certainty.
_NAME_CONTINUES = re.compile(
    rf"(?:{_NAME_WORD})|\s*\(|\s*[{_QUOTES}]?\s*:\s*[{_H}]|\s*[{_MAQAF}-][{_H}]"
)
SECTION_DEPENDENCY = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?P<proclitics>[ובלמשכה]{{0,3}})(?:סעיפים|סעיף)\s+"
    rf"(?P<section>\d+{_SECTION_SUFFIX})(?![0-9{_H}{_GERESH_MARKS}])"
    rf"(?P<fragments>(?:\s*\(\s*{OUTLINE_LABEL}\s*\)){{0,{MAX_OUTLINE_DEPTH}}})"
    # A fragment the label grammar cannot read (``(1000)``, a glued ``(x)``),
    # or one past the supported depth, leaves the citation unreadable rather
    # than shortened to its section.
    rf"(?!\s*\(\s*(?:\d+|{OUTLINE_LABEL})\s*\))(?!\()"
    rf"(?P<instrument>\s*(?:,\s*)?{_INSTRUMENT_INTRO}"
    rf"(?P<instrument_kind>{_INSTRUMENT_KIND})"
    rf"{_NOT_HEBREW_AFTER}"
    rf"(?P<instrument_name>{_INSTRUMENT_NAME}))?"
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
    # Any attached prefixes: ``שחוק``, ``כשהחוק``, ``ושבחוק``, ``מן החוק``.
    rf"{_NOT_HEBREW_BEFORE}(?:של\s+|(?:אשר\s+)?(?:מן\s+|מתוך\s+)?)?"
    rf"[ובלמשכה]{{0,4}}[{_QUOTES}]?ה?[{_QUOTES}]?"
    rf"(?P<kind>{_INSTRUMENT_KIND})"
    rf"{_NOT_HEBREW_AFTER}"
    rf"(?P<name>{_INSTRUMENT_NAME})"
)
# A law named just before a native citation: ``חוק הביטוח הלאומי, סעיף 14``.
_PREPOSED_INSTRUMENT = re.compile(
    rf"{_NOT_HEBREW_BEFORE}[ובלמשכה]{{0,4}}[{_QUOTES}]?ה?[{_QUOTES}]?"
    rf"{_INSTRUMENT_KIND}{_NOT_HEBREW_AFTER}(?P<name>{_INSTRUMENT_NAME})"
    rf"[{_QUOTES}]?\s*[,:]\s*"
)
_CURRENT_INSTRUMENT = ("current", ())
_UNRESOLVED_INSTRUMENT = ("unresolved", ())
Instrument = tuple[str, tuple[str, ...]]


def _instrument_kind(kind: str) -> str:
    return "פקודה" if kind.startswith("פקוד") else kind


def _instrument_words(name: str) -> tuple[str, ...]:
    """Name words, with a parenthetical qualifier kept as one word."""

    def word(token: str) -> str:
        return _normalize_word(token.strip(_QUOTES + _MAQAF + "-"))

    return tuple(
        "(" + " ".join(word(part) for part in token[1:-1].split()) + ")"
        if token.startswith("(")
        else word(token)
        for token in re.findall(r"\([^()]*\)|[^\s()]+", name)
        if word(token) or token.startswith("(")
    )


def _name_is_cut_short(text: str, end: int) -> bool:
    return _NAME_CONTINUES.match(text, end) is not None


def _named_instrument(
    kind: str,
    words: tuple[str, ...],
    mentions: dict[str, tuple[list[int], list[Instrument]]],
    start: int,
) -> Instrument:
    """Resolve ``לחוק זה``, ``לחוק האמור`` or ``לחוק הביטוח הלאומי``."""

    first = words[0] if words else ""
    if first in _CURRENT_INSTRUMENT_NAMES:
        return _CURRENT_INSTRUMENT
    if words and first not in _CONTEXTUAL_INSTRUMENT_NAMES:
        return (kind, words)
    # A contextual (or bare) name is the same kind of instrument last named.
    positions, instruments = mentions.get(kind, ([], []))
    index = bisect.bisect_left(positions, start) - 1
    return instruments[index] if index >= 0 else _UNRESOLVED_INSTRUMENT


def _explicit_instrument_mentions(
    text: str,
) -> dict[str, tuple[list[int], list[Instrument]]]:
    """Explicitly named instruments of each kind, in text order."""

    mentions: dict[str, tuple[list[int], list[Instrument]]] = {}
    for mention in _INSTRUMENT_MENTION.finditer(text):
        words = _instrument_words(mention.group("name"))
        if words and words[0] not in _CONTEXTUAL_INSTRUMENT_NAMES:
            kind = _instrument_kind(mention.group("kind"))
            positions, instruments = mentions.setdefault(kind, ([], []))
            positions.append(mention.start())
            # A cut-short name is the last law named, but not a known one.
            instruments.append(
                _UNRESOLVED_INSTRUMENT
                if _name_is_cut_short(text, mention.end())
                else (kind, words)
            )
    return mentions


def _section_citations(
    text: str,
) -> tuple[tuple[re.Match[str], Instrument | None], ...]:
    """Return each section citation with the instrument it belongs to.

    ``None`` is a citation that names no instrument; in a source that is the
    current law, while a reason may leave the instrument to its context.
    """

    mentions = _explicit_instrument_mentions(text)
    citations: list[tuple[re.Match[str], Instrument | None]] = []
    for match in SECTION_DEPENDENCY.finditer(text):
        kind = match.group("instrument_kind")
        if not kind:
            instrument = None
        elif _name_is_cut_short(text, match.end()):
            instrument = _UNRESOLVED_INSTRUMENT
        else:
            instrument = _named_instrument(
                _instrument_kind(kind),
                _instrument_words(match.group("instrument_name") or ""),
                mentions,
                match.start("instrument"),
            )
        citations.append((match, instrument))
    return tuple(citations)


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


@functools.lru_cache(maxsize=256)
def _clause_stops(text: str) -> tuple[int, ...]:
    return tuple(match.start() for match in re.finditer(r"[;\n]|\.(?!\d)", text))


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Bound a clause by ``;``, a line break or a sentence period, not ``5.17``."""

    stops = _clause_stops(text)
    before = bisect.bisect_left(stops, start)
    after = bisect.bisect_left(stops, end)
    clause_start = stops[before - 1] + 1 if before else 0
    clause_end = stops[after] if after < len(stops) else len(text)
    return clause_start, clause_end


# A negation, prohibition or exclusion anywhere before a source linker in its
# clause negates the link (``אין לפעול בהתאם ל``, ``נאסר ...``, ``אי אפשר
# ...``), as English's ``_source_link_scope_has_unreset_negation`` reads ``do
# not act in accordance with``. A main-clause negation governs to the end of
# its clause, relative clauses included (``אין לחשב את התשלום שנקבע בהתאם
# ל``). A negation inside a relative clause (``שאינו``, ``שלא``) ends where a
# later relative clause carrying the link begins: the linker is a participle
# (``הקבוע ב``, ``שנקבע ב``) or the word just before it is a relative verb
# (``החלק שאינו עולה על הסכום שנקבע לפי סעיף 341``, section 14(ו1)). Any other
# negation stands, so ``מבוטח שלא עבד ישלם לפי`` is read as negating:
# stricter than English, which does not check ``under``.
_NEGATIONS = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?P<relative>(?:ו)?(?:כ)?ש)?(?:[ומה]{{0,2}})"
    r"(?:אין|לא|אינ(?:ו|ה|ם|ן|י|ך|כם|כן|נו|נה|ני)|אסור(?:ה|ים|ות)?|בל|לבל|בלתי"
    r"|נאסר(?:ה|ו|ים|ות)?|איסור|מנוע(?:ה|ים|ות)?"
    r"|בלי|ללא|מבלי|למעט|מלבד|להוציא|זולת)"
    rf"{_NOT_HEBREW_AFTER}"
    rf"|{_NOT_HEBREW_BEFORE}[ושכמה]{{0,2}}אי{_SEP}(?:אפשר|ניתן){_NOT_HEBREW_AFTER}"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?(?:פרט{_SEP}?ל|חוץ{_SEP}?מ)(?=[{_H}])"
)
_RELATIVE_LINK_VERBS = (
    "נקבע נקבעה נקבעו נקבעים נקבעות הוגדר הוגדרה הוגדרו מוגדר מוגדרת מוגדרים "
    "מוגדרות נמנה נמנו מנוי מנויה מנויים מנויות קבוע קבועה קבועים קבועות חל חלה "
    "חלים חלות משתלם משתלמת משתלמים משתלמות מחושב מחושבת מחושבים מחושבות חושב "
    "חושבה חושבו נכלל נכללה נכללו ייקבע ייקבעו יחושב יחושבו"
)
_RELATIVE_VERB_BEFORE_LINKER = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?(?:ש|אשר\s+)(?:"
    + "|".join(_RELATIVE_LINK_VERBS.split())
    + rf"){_NOT_HEBREW_AFTER}\s*$"
)
_PARTICIPLE_LINKER = re.compile(r"(?:ו)?(?:ש|ה)(?:נקבע|קבוע|מנוי)")
_DISPLACEMENT = re.compile(
    r"(?:(?:מ)?בלי\s+לגרוע|על\s+אף|אין\s+ב[^.;\n]{0,80}כדי\s+לגרוע)"
    r"[^.;\n]{0,40}$"
)
_SOURCE_LINKER = re.compile(
    r"(?:"
    rf"{_NOT_HEBREW_BEFORE}(?:ו|ש|וש)?(?:לפי|על[\s{_MAQAF}-]פי|עפ[״\"]י)\s*"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?בהתאם\s+ל"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?כאמור\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?כ(?:משמעות|הגדרת)(?:ו|ה|ם|ן)?\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?כ(?:מפורט|קבוע)\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו|ש)?(?:ה)?(?:קבוע|מנוי)(?:ה|ים|ות)?\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?שנקבע(?:ו|ה)?\s+ב"
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?(?:ה)?הוראות\s+(?:לפי\s*|ש?ב)?"
    r")$"
)


def _source_linker(text: str, start: int, end: int) -> re.Match[str] | None:
    """The operative linker ending at ``end``, read from ``start``, unless it displaces.

    The text is searched in place, so a word cut by ``start`` keeps its left
    boundary (``כלפי`` is not ``לפי``).
    """

    if _DISPLACEMENT.search(text, start, end):
        return None
    return _SOURCE_LINKER.search(text, start, end)


def _link_resets(text: str, negation: re.Match[str], linker: re.Match[str]) -> bool:
    """Whether a relative clause after a relative negation carries the link."""

    if not negation.group("relative"):
        return False
    if _PARTICIPLE_LINKER.match(linker.group(0)):
        return True
    # The relative verb is adjacent to the linker, so a bounded look suffices.
    return (
        _RELATIVE_VERB_BEFORE_LINKER.search(
            text, max(negation.end(), linker.start() - 24), linker.start()
        )
        is not None
    )


_LINK_WINDOW = 240


@dataclass
class _LinkedInstruments:
    """The instruments whose citations of one section a scope makes operative."""

    current: bool = False
    names: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)
    prefixes: dict[str, set[tuple[str, ...]]] = field(default_factory=dict)

    def add(self, instrument: Instrument) -> None:
        if instrument == _CURRENT_INSTRUMENT:
            self.current = True
            return
        if instrument == _UNRESOLVED_INSTRUMENT:
            return
        kind, words = instrument
        self.names.setdefault(kind, set()).add(words)
        prefixes = self.prefixes.setdefault(kind, set())
        for length in range(1, len(words) + 1):
            prefixes.add(words[:length])

    def admits(self, instrument: Instrument) -> bool:
        """``_same_instrument`` against every entry, in time linear in the name."""

        if instrument == _CURRENT_INSTRUMENT:
            return self.current
        if instrument == _UNRESOLVED_INSTRUMENT:
            return False
        kind, words = instrument
        return words in self.prefixes.get(kind, ()) or any(
            words[:length] in self.names.get(kind, ())
            for length in range(1, len(words) + 1)
        )


@functools.lru_cache(maxsize=256)
def _linked_source_instruments(
    source_scope_text: str,
) -> dict[tuple[str, tuple[str, ...]], _LinkedInstruments]:
    """Instruments of the operative citations, by section and fragment prefix.

    Each scope citation that a linker makes operative (``לפי סעיף 345ב(א)``)
    is entered under its section with every prefix of its fragments, so a
    reason citing ``345ב`` or ``345ב(א)`` is answered by one lookup.
    """

    index: dict[tuple[str, tuple[str, ...]], _LinkedInstruments] = {}
    negations = list(_NEGATIONS.finditer(source_scope_text))
    negation_ends = [negation.end() for negation in negations]
    for cited, cited_instrument in _section_citations(source_scope_text):
        clause_start, _clause_end = _clause_bounds(
            source_scope_text, cited.start(), cited.end()
        )
        # The linkers and the displacement guard read at most ~140 characters
        # before the citation; the window keeps a long clause linear. The
        # negation index covers the whole clause.
        window_start = max(clause_start, cited.start() - _LINK_WINDOW)
        linker = _source_linker(
            source_scope_text, window_start, cited.end("proclitics")
        )
        if linker is None:
            continue
        last = bisect.bisect_right(negation_ends, linker.start()) - 1
        if (
            last >= 0
            and negations[last].start() >= clause_start
            and not _link_resets(source_scope_text, negations[last], linker)
        ):
            continue
        section, fragments = _dependency_identity(cited)
        for length in range(len(fragments) + 1):
            index.setdefault((section, fragments[:length]), _LinkedInstruments()).add(
                cited_instrument or _CURRENT_INSTRUMENT
            )
    return index


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
    linked = _linked_source_instruments(source_scope_text).get((section, fragments))
    if linked is None:
        return False
    return instrument is None or linked.admits(instrument)


# Citations an English or canonical reader counts, so a mixed-language
# clause such as ``il/statute/.../section-14(e1): חסר ...`` is read as having
# two citations, as ``_PRECISE_DEFERRAL_DEPENDENCY`` reads it.
_OTHER_CITATION = re.compile(
    # ``_PRECISE_DEFERRAL_DEPENDENCY`` in source_completeness, verbatim;
    # ``test_other_citations_use_the_english_citation_grammar`` holds them equal.
    r"(?:"
    r"§{1,2}\s*\d+[a-z]?|"
    r"\b(?:section|subsection|paragraph|absatz|abs\.|satz|nummer|nr\.)"
    r"\s*\d+[a-z]?|"
    r"\b[a-z]{2}(?:-[a-z0-9-]+)?/(?:statute|regulation|guidance|manual)/"
    r"[A-Za-z0-9_./-]+|"
    r"\b[a-z]{2}(?:-[a-z0-9-]+)?:"
    r"(?:statutes|regulations|guidance|manuals)/[A-Za-z0-9_./-]+#[A-Za-z0-9_]+"
    r")",
    flags=re.IGNORECASE,
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
            "רישום תקנה תקנות דרישה דרישות כלל כללי כללים "
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
            "של או וכן שנתי שנתית החל החלה החלים החלות מנהלי "
            "מנהלית ברוטו משק בית משפטי משפטית מקומי מקומית ציבורי ציבורית "
            "ממלכתי ממלכתית מס שנה שנת מתואם מתואמת ראשוני ראשונית"
        ).split(),
    )
)
# Only the conjunction and the article attach to a subject-phrase word; a
# preposition (``בהגשה``, ``לזכאות``, ``מהתקנות``) makes it a description that
# English's bounded phrase does not allow.
_TERM_PREFIX_LETTERS = "וה"


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
    r"^\s*(?:לא|אי)\s+(?:ניתן|אפשר)\s+(?:לחשב|לקודד)"
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


# Longest negation the prefix pattern reads (``וכלל  לא``), with room for the
# whitespace inside it; whitespace before the signal is skipped first.
_NEGATION_WINDOW = 32


def _signals(text: str) -> list[re.Match[str]]:
    """Missing-dependency signals of ``text`` that no negation governs.

    The negation is read in a bounded window before each signal, so the scan
    stays linear in the reason's length.
    """

    signals = []
    for signal in _SIGNAL.finditer(text):
        end = signal.start()
        while end and text[end - 1].isspace():
            end -= 1
        if not _NEGATED_SIGNAL_PREFIX.search(text, max(0, end - _NEGATION_WINDOW), end):
            signals.append(signal)
    return signals


def _missing_state_follows(after: str) -> bool:
    """``…סעיף 345ב טרם קודד`` / ``…סעיף 345ב אינו מקודד`` and nothing more.

    The state follows the citation directly, as English reads ``section 345b
    is not yet encoded``; English refuses ``, which is not encoded``, so the
    relative ``, שטרם קודד`` is refused too.
    """

    state = re.match(rf"\s*{_MISSING_STATE}{_NOT_HEBREW_AFTER}", after)
    return state is not None and not after[state.end() :].strip(" \t,.)")


def _until_state_follows(after: str) -> bool:
    state = re.match(rf"\s*{_UNTIL_STATE}{_NOT_HEBREW_AFTER}", after)
    return state is not None and not after[state.end() :].strip(" \t,.)")


class _ReasonIndex:
    """Signals, adversatives and citations of one reason, found once.

    Each dependency then looks up what precedes it in its clause, which keeps
    a reason with many citations linear rather than rescanning every prefix.
    """

    def __init__(
        self,
        reason: str,
        other_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
        masked_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
    ) -> None:
        self.signals = _signals(reason)
        self.signal_ends = [signal.end() for signal in self.signals]
        self.adversatives = [match.start() for match in _ADVERSATIVE.finditer(reason)]
        # A citation starts at its ``סעיף`` word: its prefix letters may belong
        # to a signal (``תלוי בסעיף``), and the citation is still between that
        # signal and a later one.
        other_spans = sorted(
            (match.start(), match.end())
            for match in (other_citations or _OTHER_CITATION.finditer)(reason)
        )
        spans = sorted(
            [
                (match.end("proclitics"), match.end())
                for match, _ in _section_citations(reason)
            ]
            + other_spans
        )
        self.citation_starts = [span_start for span_start, _ in spans]
        # The earliest end among the spans from each index on.
        self.earliest_end_from = [0] * len(spans)
        earliest = len(reason) + 1
        for position in range(len(spans) - 1, -1, -1):
            earliest = min(earliest, spans[position][1])
            self.earliest_end_from[position] = earliest
        # Clause stops, but not the periods inside the citations English masks
        # (``42 U.S.C. 9``, ``R.S. 47:295``), as ``_reason_clause_bounds`` does.
        masked_spans = sorted(
            (match.start(), match.end())
            for match in (masked_citations(reason) if masked_citations else ())
        )
        self.stops = []
        span_index = 0
        for stop in _clause_stops(reason):
            while (
                span_index < len(masked_spans) and masked_spans[span_index][1] <= stop
            ):
                span_index += 1
            inside = (
                span_index < len(masked_spans)
                and masked_spans[span_index][0] <= stop < masked_spans[span_index][1]
            )
            if not inside:
                self.stops.append(stop)
        self.length = len(reason)

    def clause_bounds(self, start: int, end: int) -> tuple[int, int]:
        before = bisect.bisect_left(self.stops, start)
        after = bisect.bisect_left(self.stops, end)
        clause_start = self.stops[before - 1] + 1 if before else 0
        clause_end = self.stops[after] if after < len(self.stops) else self.length
        return clause_start, clause_end

    def last_signal(self, start: int, end: int) -> re.Match[str] | None:
        """The last unnegated signal that starts at ``start`` or later and ends by ``end``."""

        index = bisect.bisect_right(self.signal_ends, end) - 1
        if index < 0 or self.signals[index].start() < start:
            return None
        return self.signals[index]

    def has_adversative(self, start: int, end: int) -> bool:
        index = bisect.bisect_left(self.adversatives, start)
        return index < len(self.adversatives) and self.adversatives[index] < end

    def has_citation(self, start: int, end: int) -> bool:
        """Whether a citation lies wholly inside ``[start, end)``."""

        first = bisect.bisect_left(self.citation_starts, start)
        return (
            first < len(self.citation_starts) and self.earliest_end_from[first] <= end
        )


def reason_names_source_bound_dependency(
    reason: str,
    source_scope_text: str,
    *,
    corpus_citation_path: str,
    other_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
    masked_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
) -> bool:
    """Accept a Hebrew reason naming a missing external section the scope applies.

    The Hebrew form of the English external-dependency route, and no looser
    than it on the same text. The cited section must differ from the current
    one (or belong to another instrument), a missing-dependency signal must
    govern it within one clause with no negation or other citation in
    between, neither that clause nor a following citation-free clause may
    hold an adversative, and the deferred source scope must make that section
    of that instrument operative (``לפי סעיף 345ב``, ``כהגדרתה בסעיף 345ב``).
    As in English, only the missing and without routes read a citation
    qualified by its law (``סעיף 345ב לחוק הביטוח הלאומי``); the requires,
    until and cannot-compute routes need the bare section with its state.
    The instrument comparison is stricter than the English prose route.
    ``other_citations`` is the English reader's citation finder (U.S. Code,
    R.S., ``§``, ``Absatz`` and canonical paths); without it the English
    precise-dependency grammar alone stands in. ``masked_citations`` are the
    citations whose periods the English clause reader masks.
    """

    current_section = native_section_label(corpus_citation_path)
    if current_section is None or not source_scope_text.strip():
        return False
    index = _ReasonIndex(reason, other_citations, masked_citations)
    stops = index.stops
    for dependency, instrument in _section_citations(reason):
        if not _dependency_is_external(dependency, instrument, current_section):
            continue
        clause_start, clause_end = index.clause_bounds(
            dependency.start(), dependency.end()
        )
        if index.has_adversative(clause_start, clause_end):
            continue
        # English refuses a reason whose next clause, citing nothing, reverses
        # it (``חסרה הוראת סעיף 345ב. אולם היא זמינה.``).
        if clause_end < len(reason):
            following = bisect.bisect_right(stops, clause_end)
            following_end = stops[following] if following < len(stops) else len(reason)
            if index.has_adversative(
                clause_end + 1, following_end
            ) and not index.has_citation(clause_end + 1, following_end):
                continue
        # A signal may be glued to the citation's prefix letters, as in
        # ``תלוי בסעיף`` (depends on section) or ``עד שסעיף`` (until section).
        reference_start = dependency.end("proclitics")
        signal = index.last_signal(clause_start, reference_start)
        if signal is None:
            continue
        bridge = reason[signal.end() : reference_start]
        if len(bridge) > 240 or index.has_citation(signal.end(), reference_start):
            continue
        after = reason[dependency.end() : clause_end]
        qualified = bool(dependency.group("instrument"))
        if signal.group("missing") or signal.group("without"):
            # As in English, an earlier citation in the same clause leaves the
            # missing claim ambiguous between the two.
            names_missing = not index.has_citation(clause_start, signal.start())
        elif signal.group("requires"):
            # ``החישוב שתלוי בסעיף 345ב טרם קודד`` or ``החישוב התלוי ...`` says
            # the calculation that depends on the section is unencoded, not the
            # section: a relative or attributive prefix makes the state belong to
            # the subject, so only a bare or ``ו`` signal opens this route.
            names_missing = (
                not qualified
                and re.match(r"(?:ו)?(?:דורש|תלוי)", signal.group(0)) is not None
                and _requirement_subject_is_bounded(
                    reason[clause_start : signal.start()]
                )
                and _missing_state_follows(after)
            )
        elif signal.group("until"):
            names_missing = not qualified and (
                (
                    signal.group("until_verb") is not None
                    and _introduction_is_bounded(bridge)
                    and not after.strip(" \t,.)")
                )
                or (
                    signal.group("until_verb") is None
                    and not bridge.strip()
                    and _until_state_follows(after)
                )
            )
        else:
            # ``לא ניתן לחשב`` only introduces; the citation must carry its
            # own missing state (``כי סעיף 345ב טרם קודד``).
            names_missing = (
                not qualified
                and re.fullmatch(
                    r"\s*(?:את\s+[^.;\n]{1,80}?\s+)?"
                    r"(?:כי|מאחר\s+ש|משום\s+ש|כיוון\s+ש)?\s*",
                    bridge,
                )
                is not None
                and _missing_state_follows(after)
            )
        if names_missing and _source_scope_links_dependency(
            source_scope_text, dependency, instrument
        ):
            return True
    return False


# Counterparts of ``_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE``, word for word. They
# only choose the diagnostic for a reason that attempts a runtime gap on a
# branch stating a computation; the runtime-gap route itself counts English
# words and so admits no Hebrew reason. The words:
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
