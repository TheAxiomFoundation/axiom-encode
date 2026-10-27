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
import itertools
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from axiom_encode.harness._unicode_default_ignorables import DEFAULT_IGNORABLE_RANGES
from axiom_encode.harness.hebrew_prefixes import (
    PREFIX_BOUNDARY,
    PREFIX_JOINT,
    prefix_chain,
    prefix_word,
)

HEBREW_LETTER_CLASS = "א-ת"
_H = HEBREW_LETTER_CLASS

# Marks that change no word, chosen by Unicode property rather than listed:
#
# - every format character (General_Category ``Cf``): the bidirectional
#   marks, embeddings and isolates, the zero-width spaces and joiners, the
#   soft hyphen, the invisible operators U+2061-U+2064, U+180E and the tags;
# - the Hebrew points and cantillation, category ``Mn`` in U+0591-U+05C7 (the
#   maqaf, paseq, sof pasuq and inverted nun there are punctuation and stay);
# - Unicode's ``Default_Ignorable_Code_Point`` property, including the
#   unassigned code points it reserves for invisible characters. The checked-in
#   range table comes from DerivedCoreProperties.txt for Python's Unicode
#   version; scripts/generate_default_ignorable_table.py regenerates it;
# - the surrogates (category ``Cs``), which are no characters: YAML decodes an
#   escaped astral mark (``"\ud804\udcbd"``, U+110BD) into two of them, so a
#   reason written with one would otherwise hide the word around it.
#
# Combining marks of other scripts are parts of letters (a German ``u`` with
# U+0308 is ``ü``) and stay. ``לא`` with a holam (U+05B9), a right-to-left mark
# (U+200F) or an invisible separator (U+2063) inside it is ``לא`` to a reader,
# so the guards read through all of them.
_HEBREW_MARK_BLOCK = (0x0591, 0x05C7)
# Precomposed pointed letters (U+FB4B, vav with holam) read as their letters.
_PRESENTATION_FORMS = (0xFB1D, 0xFB4F)


@functools.cache
def folded_marks() -> frozenset[int]:
    """The code points ``fold_marks`` drops, read from the Unicode database.

    The database is read once, on first use, at C speed: the categories of
    all code points map in one pass. Default ignorables come from the Unicode
    property table, since ``unicodedata`` does not expose that property.
    """

    codes = range(sys.maxunicode + 1)
    categories = list(map(unicodedata.category, map(chr, codes)))
    format_and_surrogates = itertools.compress(
        codes, map(frozenset(("Cf", "Cs")).__contains__, categories)
    )
    hebrew_marks = (
        code
        for code in range(_HEBREW_MARK_BLOCK[0], _HEBREW_MARK_BLOCK[1] + 1)
        if categories[code] == "Mn"
    )
    default_ignorables = itertools.chain.from_iterable(
        range(low, high + 1) for low, high in DEFAULT_IGNORABLE_RANGES
    )
    return frozenset(
        itertools.chain(format_and_surrogates, hebrew_marks, default_ignorables)
    )


def _code_ranges(codes: Iterable[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for code in sorted(codes):
        if ranges and ranges[-1][1] == code - 1:
            ranges[-1] = (ranges[-1][0], code)
        else:
            ranges.append((code, code))
    return ranges


@functools.cache
def _fold_rules() -> tuple[dict[int, str | None], re.Pattern[str]]:
    """The translation table and the pattern of every character it changes."""

    table: dict[int, str | None] = dict.fromkeys(folded_marks())
    for code in range(_PRESENTATION_FORMS[0], _PRESENTATION_FORMS[1] + 1):
        char = chr(code)
        if unicodedata.category(char) == "Cn":
            continue
        table[code] = "".join(
            part
            for part in unicodedata.normalize("NFKD", char)
            if not unicodedata.combining(part)
        )
    foldable = re.compile(
        "["
        + "".join(
            re.escape(chr(low)) + (f"-{re.escape(chr(high))}" if high > low else "")
            for low, high in _code_ranges(table)
        )
        + "]"
    )
    return table, foldable


# The Hebrew block from its first mark, and the presentation forms.
_HEBREW_SCRIPT = re.compile("[\u0591-\u05f4\ufb1d-\ufb4f]")


def fold_marks(text: str) -> str:
    """Drop the marks that change no word from text in Hebrew script.

    Every negation, adversative, signal, citation and outline guard reads the
    folded copy, so a pointed or direction-marked word is the word itself.
    Text with no Hebrew script is returned as is, the same object, so no
    English or German reading changes. A reader that reports offsets reads a
    :class:`FoldedText`, which maps each folded offset back to the original.
    """

    if _HEBREW_SCRIPT.search(text) is None:
        return text
    table, foldable = _fold_rules()
    if foldable.search(text) is None:
        return text
    return text.translate(table)


class FoldedText:
    """``fold_marks(original)``, with its offsets mapped back to ``original``.

    The map records only the folded characters, so building it costs one
    scan of the marks and no work per unmarked character. A folded offset
    of a kept character maps to that character; a dropped mark belongs to
    no folded offset, so a slice between two mapped starts holds the marks
    between them; a presentation form that folds to two letters (U+FB4F)
    maps both to itself.
    """

    __slots__ = ("original", "text", "_ends", "_shifts", "_sources", "_starts")

    def __init__(self, original: str, *, force: bool = False) -> None:
        """Fold ``original``; ``force`` folds it even with no Hebrew script.

        A forced fold is for a window cut from a longer text, whose Hebrew
        letters may lie past the window: a reader that needs a Hebrew letter
        in what it matches reads the window as the whole text would fold it.
        """

        self.original = original
        if force:
            table, foldable = _fold_rules()
            self.text = (
                original.translate(table) if foldable.search(original) else original
            )
        else:
            self.text = fold_marks(original)
        # For each folded character: where its replacement starts and ends in
        # the folded text, the original offset it came from, and the shift of
        # the kept text after it.
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._sources: list[int] = []
        self._shifts: list[int] = [0]
        if self.text is original:
            return
        table, foldable = _fold_rules()
        shift = 0
        for mark in foldable.finditer(original):
            replacement = table[ord(mark.group(0))] or ""
            start = mark.start() - shift
            self._starts.append(start)
            self._ends.append(start + len(replacement))
            self._sources.append(mark.start())
            shift += 1 - len(replacement)
            self._shifts.append(shift)

    def original_start(self, offset: int) -> int:
        """The original offset of the folded character at ``offset``."""

        index = bisect.bisect_right(self._ends, offset)
        if index < len(self._starts) and self._starts[index] <= offset:
            return self._sources[index]
        return offset + self._shifts[index]

    def original_end(self, offset: int) -> int:
        """The original offset just past the folded character before ``offset``."""

        return self.original_start(offset - 1) + 1 if offset > 0 else 0


# Where a Hebrew letter meets a letter of another script. A Hebrew prefix
# glues to a word of another script (``לsection 2``, ``וnot``, ``שohne``), and
# Python's ``\b`` sees one word there, so ``\bnot\b`` misses the ``not`` of
# ``וnot``. A script change is a word boundary for the readers of negations,
# exclusions and adversatives: they read ``_split_scripts``'s copy, with a
# space at each junction (#1882). The readers of signals, links and citations
# do not, so a glued word is read only where it refuses.
_SCRIPT_JUNCTION = re.compile(
    rf"(?<=[{_H}])(?=[^\W\d_\u0590-\u05ff])|(?<=[^\W\d_\u0590-\u05ff])(?=[{_H}])"
)


def _split_scripts(text: str) -> tuple[str, list[int]]:
    """``text`` with a space at each junction of scripts, and the junctions.

    A junction is the offset in ``text`` that the space is inserted before.
    Text with no junction is returned as is, the same object.
    """

    junctions = [match.start() for match in _SCRIPT_JUNCTION.finditer(text)]
    if not junctions:
        return text, junctions
    bounds = [0, *junctions, len(text)]
    return (
        " ".join(text[low:high] for low, high in itertools.pairwise(bounds)),
        junctions,
    )


def _split_starts(
    text: str, finders: Iterable[Callable[[str], Iterable[re.Match[str]]]]
) -> list[int]:
    """Where the finders' matches start in ``text``, read with scripts split.

    Each finder reads ``_split_scripts(text)``; an offset there maps back past
    the spaces inserted before it, and one that falls on an inserted space
    maps to the character after it. The starts are sorted.
    """

    view, junctions = _split_scripts(text)
    starts = [match.start() for finder in finders for match in finder(view)]
    if junctions:
        # The k-th inserted space stands at view offset junctions[k] + k.
        inserted = [junction + index for index, junction in enumerate(junctions)]
        starts = [start - bisect.bisect_left(inserted, start) for start in starts]
    return sorted(starts)


def cross_language_text(text: str) -> str | None:
    """``text`` as the shared checker's English guards read it with Hebrew in it.

    ``None`` for text with no Hebrew script, which keeps its English or German
    reading. Otherwise the marks are folded and a space stands at each
    junction of scripts, so ``לא`` with U+200F inside it, ``וnot`` and
    ``notמחושב`` show their words. A guard that reads this copy as well as
    the text, and refuses when either refuses, reads no text more loosely.
    """

    if _HEBREW_SCRIPT.search(text) is None:
        return None
    return _split_scripts(fold_marks(text))[0]


# Geresh and gershayim, and the marks that stand in for them: ASCII quotes and
# the typographic quotes a word processor substitutes (``345י”א``). A numeral
# read without them would lose its last letters and name another section.
_GERESH_MARKS = "׳״'\"“”‘’"
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


# The closing parenthesis ends a marker, so a marker may touch its text
# (``(ב)המוסד חייב``), as extracted statutes sometimes print it.
_OUTLINE_CHAIN = re.compile(
    rf"(?m)^[ \t]*(?P<chain>\({OUTLINE_LABEL}\)(?:[ \t]*\({OUTLINE_LABEL}\))*)"
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

    Labels are read on the folded text (:class:`FoldedText`), so a pointed or
    direction-marked label such as ``(א‏)`` or ``(אֹ)`` is ``(א)``: its marker
    is the folded label, and its start is mapped back to an offset into
    ``source_text``, so every branch slice and message reads the original.
    """

    if not HEBREW_LETTER.search(source_text):
        # Keep non-Hebrew sources at one C-speed scan: a CFR-style outline of
        # 20,000 `(n)` lines would otherwise parse every label here.
        return ()
    folded = FoldedText(source_text)
    chains = []
    for match in _OUTLINE_CHAIN.finditer(folded.text):
        labels = [
            parse_outline_label(text)
            for text in _CHAIN_LABEL.findall(match.group("chain"))
        ]
        if labels and all(label is not None for label in labels):
            chains.append((folded.original_start(match.start("chain")), labels))
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


def _leading_chain_match(text: str) -> tuple[FoldedText, re.Match[str] | None]:
    """Match ``LEADING_OUTLINE_CHAIN`` on the folded start of ``text``.

    Only the start is folded: a window that grows fourfold until the chain
    match is decided, so a call costs the chain's own length, marks
    included, however long the text after it.
    """

    width = 64
    while True:
        window = FoldedText(text[:width], force=True)
        match = LEADING_OUTLINE_CHAIN.match(window.text)
        if width >= len(text):
            return window, match
        rest = window.text[match.end() :] if match else window.text.lstrip()
        # Decided once the window holds a character that ends the chain: one
        # other than an opening parenthesis, or a label's closing one.
        if (match is None or match.end() < len(window.text)) and (
            rest[:1] not in {"", "("} or ")" in rest
        ):
            return window, match
        width *= 4


def strip_leading_outline_chain(text: str) -> str | None:
    """Strip a leading Hebrew marker chain such as ``(ב) (1)``; else ``None``.

    The chain is read on the folded text, as ``hebrew_outline_markers`` reads
    it, and the original text after it is returned.
    """

    folded, match = _leading_chain_match(text)
    if match is None or not HEBREW_LETTER.search(match.group("chain")):
        return None
    labels = [
        parse_outline_label(label)
        for label in _CHAIN_LABEL.findall(match.group("chain"))
    ]
    if not labels or any(label is None for label in labels):
        return None
    return text[folded.original_end(match.end()) :].lstrip()


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
    # A section suffix is one or two letters (``121ב``, ``345יא``).
    value = ordinal_path_value(suffix) if len(suffix) <= 2 else None
    return number + hebrew_numeral(value) if value and value <= 99 else None


def native_path_label(component: str) -> str | None:
    """Render one path component as its printed Hebrew label: ``e1`` -> ``ה1``."""

    # Labels are small numerals (``OUTLINE_LABEL``): at most two path letters,
    # checked before any value or Hebrew numeral is built from them.
    number = re.fullmatch(r"(\d+)([a-z]{0,2})", component)
    if number is not None:
        if not number.group(2):
            return number.group(1)
        value = ordinal_path_value(number.group(2))
        if value is None or value > _MAX_LETTER_LABEL_VALUE:
            return None
        return number.group(1) + hebrew_numeral(value)
    letters = re.fullmatch(r"([a-z]{1,2})(\d*)", component)
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
    # ``לאותו חוק``, ``לאותה פקודה``: "that law", a contextual reference.
    rf"(?:אות(?:ו|ה|ם|ן)\s+)?[{_QUOTES}]?ה?[{_QUOTES}]?"
)
_KIND_PATTERNS = {"חוק": "חוק", "פקודה": "פקוד[הת]", "תקנות": "תקנות", "צו": "צו"}


def current_instrument_kind(corpus_citation_path: str) -> str:
    """The kind of the current instrument: a law, unless its slug says otherwise."""

    parts = corpus_citation_path.lower().strip("/").split("/")
    slug = parts[2] if len(parts) > 2 else ""
    if "ordinance" in slug:
        return "פקודה"
    if "regulation" in slug or parts[1:2] == ["regulation"]:
        return "תקנות"
    if re.search(r"(?<![a-z])order(?![a-z])", slug):
        return "צו"
    return "חוק"


@functools.lru_cache(maxsize=64)
def _other_instrument_qualifier(current_kind: str) -> str:
    """Refuse a citation qualified by another instrument: another kind, or this
    kind under any name but ``זה``/``זו``/``אלה`` (``לפקודה זו`` is no law's)."""

    current = _KIND_PATTERNS[current_kind]
    return (
        rf"(?!\s*(?:,\s*)?{_INSTRUMENT_INTRO}"
        rf"(?:(?!{current}{_NOT_HEBREW_AFTER}){_INSTRUMENT_KIND}{_NOT_HEBREW_AFTER}"
        rf"|{current}{_NOT_HEBREW_AFTER}"
        rf"(?!\s+(?:זה|זו|הזה|הזאת|אלה|האלה){_NOT_HEBREW_AFTER})))"
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
        rf"(?!\s*\()(?![A-Za-z0-9_{_H}])"
        rf"{_other_instrument_qualifier(current_instrument_kind(corpus_citation_path))}"
        rf"{end_guard}"
    )


def _clause_start(stops: Sequence[int], offset: int) -> int:
    """Where the clause holding ``offset`` begins, given its text's sorted stops."""

    before = bisect.bisect_left(stops, offset)
    return stops[before - 1] + 1 if before else 0


class _PreposedInstruments:
    """Where one text names instruments other than the current one.

    ``חוק הביטוח הלאומי, התשנ״ה–1995, סעיף 14(ב)`` cites that law's section
    14(b), whatever dates, editions, descriptions or line breaks sit between.
    Only ``חוק זה`` ("this law", of the current kind) keeps the current
    instrument. A scan builds one index, on its first question, and asks it
    for each citation, so no citation rescans or rehashes the text.
    """

    def __init__(self, text: str, current_kind: str = "חוק") -> None:
        self._text = text
        self._current_kind = current_kind
        self._stops: list[int] | None = None
        self._mentions: list[int] = []

    def precede(self, start: int) -> bool:
        """Whether another instrument is named earlier in the clause of ``start``."""

        if self._stops is None:
            self._stops = [
                match.start() for match in _OWNERSHIP_STOP.finditer(self._text)
            ]
            self._mentions = [
                mention.start()
                for mention in _INSTRUMENT_MENTION.finditer(self._text)
                if not (
                    _instrument_kind(mention.group("kind")) == self._current_kind
                    and _instrument_words(mention.group("name"))[:1]
                    and _instrument_words(mention.group("name"))[0]
                    in _CURRENT_INSTRUMENT_NAMES
                )
            ]
        first = bisect.bisect_left(self._mentions, _clause_start(self._stops, start))
        return first < len(self._mentions) and self._mentions[first] < start


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
    if pattern is None:
        return False
    text = fold_marks(text)
    preposed = _PreposedInstruments(text, current_instrument_kind(corpus_citation_path))
    return any(not preposed.precede(match.start()) for match in pattern.finditer(text))


def native_source_reference_paths(
    source: str, corpus_citation_path: str
) -> set[tuple[str, ...]]:
    """Return branch paths a rule ``source:`` claims through native citations."""

    section = native_section_label(corpus_citation_path)
    if section is None:
        return set()
    source = fold_marks(source)
    current_kind = current_instrument_kind(corpus_citation_path)
    preposed = _PreposedInstruments(source, current_kind)
    paths: set[tuple[str, ...]] = set()
    for match in re.finditer(
        rf"{_NOT_HEBREW_BEFORE}{_PROCLITICS}{_SECTION_WORD}\s*{_native_section_pattern(section)}"
        rf"(?P<chain>(?:\s*\(\s*{OUTLINE_LABEL}\s*\)){{1,{MAX_OUTLINE_DEPTH}}})"
        rf"(?!\s*\()(?![A-Za-z0-9_{_H}])"
        rf"{_other_instrument_qualifier(current_kind)}",
        source,
    ):
        if preposed.precede(match.start()):
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
    if not section:
        return False
    source = fold_marks(source)
    current_kind = current_instrument_kind(corpus_citation_path)
    preposed = _PreposedInstruments(source, current_kind)
    return any(
        not preposed.precede(match.start())
        for match in re.finditer(
            rf"{_NOT_HEBREW_BEFORE}{_PROCLITICS}{_SECTION_WORD}\s*{_native_section_pattern(section)}"
            rf"(?![0-9A-Za-z{_H}{_GERESH_MARKS}])(?:\s*\(\s*{OUTLINE_LABEL}\s*\)){{0,{MAX_OUTLINE_DEPTH}}}(?!\s*\()"
            rf"{_other_instrument_qualifier(current_kind)}",
            source,
        )
    )


# --- Deferral precision -----------------------------------------------------

_SIGNAL_PROCLITICS = r"(?:וש|שה|וה|ו|ש|ה)?"
# A missing signal before its object: missing, not yet encoded and
# unavailable (``בלתי זמין``), the English signal words. Plain "not encoded"
# and "not available" are not English signals, so ``אינו מקודד`` and ``אינו
# זמין`` are not either. With the article, the feminine forms are the noun
# "subtraction" (``החסרה של הסכום``, plural ``החסרות``), so they are not read.
_MISSING_SIGNAL = (
    r"(?:חסר(?:ים)?|(?<!ה)חסר(?:ה|ות)"
    r"|(?:טרם|עדיין\s+(?:לא|אינו|אינה|אינם|אינן))\s+"
    r"(?:קודד(?:ה|ו)?|מקודד(?:ת|ים|ות)?)"
    r"|בלתי\s+(?:זמין|זמינ(?:ה|ים|ות)))"
)
_UNTIL_STATE = (
    r"(?:יקודד(?:ו)?|תקודד(?:נה)?|ייקבע(?:ו)?|תיקבע"
    r"|יהיה\s+זמין|תהיה\s+זמינה|יהיו\s+זמינ(?:ים|ות))"
)

# The Hebrew counterparts of ``_MISSING_DEPENDENCY_LANGUAGE``: missing,
# requires/depends on, until, without, not yet encoded, unavailable and
# cannot be computed. A one-letter conjunction, relative or article prefix
# may attach (``וחסרים``, ``שחסרה``); the next letter must end the word.
# This only says a reason speaks of a missing dependency. Of the signals
# ``_SIGNAL`` classifies, only missing and without can make a Hebrew reason
# precise: the requires, until and cannot-compute routes rest on English
# vocabulary lists (bounded subjects, ``computed``/``encoded``) that a
# Hebrew word does not translate into reliably, so they stay closed.
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
# A negation governs a signal through up to two copulas or adverbs (``לא היה
# חסר``, ``לא בהכרח חסרה``), as English reads "was not missing" and "is not
# necessarily missing".
_NEGATION_BRIDGE_WORDS = (
    "היה היתה הייתה היו יהיה תהיה יהיו להיות ממש באמת בהכרח תמיד כלל בכלל עוד "
    "אמור אמורה אמורים אמורות צפוי צפויה צפויים צפויות עשוי עשויה עשויים עשויות "
    "יכול יכולה יכולים יכולות צריך צריכה צריכים צריכות "
    # Truth claims: ``אין זה נכון ש``, ``אין בסיס לטענה ש``, ``אין ראיה ש``.
    "זה נכון בסיס לטענה ראיה"
)
# Words the negation may reach across: the negation itself and its bridge.
_NEGATION_REACH_WORDS = 6
# A space, maqaf or hyphen separates the negation from what it governs.
_NEGATION_GAP = rf"[\s{_MAQAF}-]"
# The prefixes a negation takes, in their order: the conjunction ``ו``, then a
# subordinator ``ש``, ``כש``, ``מש``, ``לכש`` or ``בש`` (``שלא``, ``משלא``
# "since ... not", ``וכשלא`` "and when ... not"), or the Aramaic relative
# ``ד`` that stands where ``ש`` does (``דלא``, ``ודלא`` "and that ... not").
# A preposition alone does not prefix a negation here, so ``מלא`` ("full"),
# ``כלא`` ("prison"), ``בטרם`` ("before") and ``ללא`` ("without", a signal of
# its own) are no negations, and neither is the rhetorical ``הלא``
# ("surely"); ``test_each_prefix_of_a_negation_is_read_or_excluded`` lists
# every prefix letter before ``לא`` and ``אין`` and why it is read or not.
_NEGATION_PREFIX = r"ו?(?:(?:מ|כ|לכ|ב)?ש|ד)?"
# ``בלתי`` ("not", "un-") is no other word after any chain of the prefix
# letters: the conjunction ``ו``, the article ``ה``, the prepositions ``ב``,
# ``כ``, ``ל`` and ``מ``, the relative ``ש`` and the Aramaic ``ד`` (``הבלתי``
# "the un-", ``לבלתי`` "so as not to", ``ומבלתי``). So it is a negation
# under every chain, as ``לא`` is under ``_NEGATION_PREFIX``.
_PREFIX_LETTERS = "ובהכלמשד"
_PREFIXED_BILTI = rf"[{_PREFIX_LETTERS}]*בלתי"
_NEGATED_SIGNAL_PREFIX = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?:{_NEGATION_PREFIX}"
    r"(?:לא|אין|אינ(?:ו|ה|ם|ן|י|ך|כם|כן|נו|נה|ני)|כלל\s+לא)"
    rf"|{_PREFIXED_BILTI})"
    rf"(?:{_NEGATION_GAP}+(?:"
    + "|".join(_NEGATION_BRIDGE_WORDS.split())
    + rf"){_NOT_HEBREW_AFTER}){{0,4}}"
    rf"{_NEGATION_GAP}*$"
)
# A negation also governs a signal from anywhere earlier in its clause, through
# an assertion, evidence or epistemic complement however worded (``לא ניתן
# לומר שחסרה``, ``אין כל ראיה לכך שחסרה``, ``טרם הוכח שחסרה``, ``לאו דווקא
# חסרה``), and a negation between a signal and its citation leaves the claim
# unread (``חסר מידע, לא ברור שחסרה הוראת סעיף 345ב``). A negation inside a
# signal (``טרם קודד``, ``לא ניתן לחשב``, ``בלתי זמין``) is that signal's own.
# ``בלתי`` ("un-", ``בלתי נכון ש`` "untrue that") negates as ``לא`` does,
# under any chain of prefix letters (``הטענה הבלתי נכונה ש``).
_REASON_NEGATION = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?:{_NEGATION_PREFIX}"
    r"(?:לא|אין|אינ(?:ו|ה|ם|ן|י|ך|כם|כן|נו|נה|ני)|טרם|מעולם|לאו)"
    rf"|{_PREFIXED_BILTI}){_NOT_HEBREW_AFTER}"
    rf"|{_NOT_HEBREW_BEFORE}{_NEGATION_PREFIX}אי[\s{_MAQAF}-]+(?:אפשר|ניתן)"
    rf"{_NOT_HEBREW_AFTER}"
)
# The English and German negators: ``_NEGATION_LANGUAGE`` in
# source_completeness, verbatim, which the shared checker passes in as
# ``other_negations``; ``test_reason_negations_use_the_shared_vocabulary``
# holds them equal. Each negates a Hebrew signal or link as ``לא`` does, so
# ``not חסר סעיף 2`` and ``Es ist nicht so, dass חסר סעיף 2`` are read as
# their translations ("section 2 is not missing") are.
_OTHER_NEGATION = re.compile(
    r"\b(?:"
    r"not|no|never|neither|nor|nothing|nowhere|cannot|(?:false|untrue)\s+that|"
    r"nicht|nichts|kein(?:e|en|em|er|es)?|keinesfalls|keineswegs|nie|niemals|"
    r"weder|nirgends|(?:falsch|unwahr),?\s+dass"
    r")\b|n['’]t\b",
    flags=re.IGNORECASE,
)
# The English and German negations, prohibitions and exclusions that negate an
# ``in accordance with`` link: ``_SOURCE_LINK_NEGATION_LANGUAGE`` in
# source_completeness, verbatim, which the shared checker passes in as
# ``other_link_negations``; ``test_link_negations_use_the_shared_vocabulary``
# holds them equal. With ``_OTHER_NEGATION`` they negate a Hebrew source
# link, so ``(א) prohibited מחושב בהתאם לסעיף 2`` links nothing, as its
# translation does not. ``without`` and ``ohne`` before a Hebrew word exclude
# what follows, as ``בלי`` does (``משולם ohne חישוב בהתאם לסעיף 2``).
_OTHER_LINK_NEGATION = re.compile(
    r"\b(?:not|never|neither|nor|cannot|nowhere|prohibit\w*|barred|"
    r"forbidden|precluded|prevented|excluded|denied|except)\b|"
    r"\b(?:denies?|rejects?|disclaims?)\b|"
    r"\b(?:false|untrue|impossible)\s+that\b|"
    r"(?:^|[,;:]\s*|\b(?:and|but|or|that|while|whereas)\s+)"
    r"no\s+(?:amount|tax|liability|computation|calculation|"
    r"determination|rate|base|statute|rule|provision|requirement)\b"
    r"(?:\s+[A-Za-z-]+){0,3}\s+(?:is|are|was|were|shall|may|must|will|"
    r"would|can|could|does|do|did|states?|provides?|requires?|"
    r"determines?|calculates?|computes?|applies?|arises?)\b|"
    r"\b(?:ca|do|does|did|is|are|was|were|"
    r"shall|will|would|could|should|must|need)n['’]t\b|"
    r"\b(?:fails?|refuses?|unable)\s+to\b|"
    r"\bwithout\s+(?:being\s+)?(?:determin\w*|comput\w*|calculat\w*|"
    r"appl\w*|impos\w*|aris\w*)\b|"
    r"\b(?:in\s+no\s+(?:case|event)|under\s+no\s+circumstances|"
    r"at\s+no\s+time|by\s+no\s+means|in\s+no\s+way|"
    r"on\s+no\s+account|under\s+no\s+conditions?|"
    r"in\s+no\s+(?:manner|respect))\b|"
    r"\b(?:rather|other)\s+than\b|"
    r"\b(?:nicht|nie|niemals|weder|nirgends|nirgendwo|kein(?:e|en|em|er|es)?|"
    r"keinesfalls|keineswegs|au(?:ß|ss)er|ausgenommen|abgesehen\s+von|"
    r"verbiet\w*|verboten\w*|untersag\w*|verwehrt\w*|ausgeschlossen\w*|"
    r"verhindert\w*|verweigert\w*|abgelehnt\w*|bestreit\w*|verneint\w*|"
    r"zurückgewiesen\w*|versäumt\w*|unterl(?:ä|a)sst|unterlassen\w*|weigert\w*|"
    r"au(?:ß|ss)erstande|(?:an)?statt|anders\s+als|"
    r"(?:falsch|unwahr|unmöglich),?\s+dass)\b|"
    r"\b(?:without|ohne)\s+(?:being\s+|zu\s+)?(?:"
    r"determin\w*|comput\w*|calculat\w*|appl\w*|impos\w*|aris\w*|"
    r"bestimm\w*|ermitt?el\w*|ermittl\w*|festgesetzt\w*|festsetz\w*|"
    r"festzusetzen|festgestellt\w*|feststell\w*|festzustellen|berechn\w*|"
    r"errechn\w*|anwend\w*|angewendet\w*|angewandt\w*|anzuwenden|erheb\w*|"
    r"erhoben\w*|auferleg\w*|aufzuerlegen|entsteh\w*|entstanden\w*)\b|"
    r"\b(?:without|ohne)\s+(?:being\s+|zu\s+)?(?=[א-ת])",
    flags=re.IGNORECASE,
)

# Words of a compound connective are separated by a space, a maqaf or a hyphen.
_SEP = rf"(?:\s+|\s*[{_MAQAF}-]\s*)"


def _connective(words: str) -> str:
    return _SEP.join(re.escape(word) for word in words.split())


# Counterparts of ``_ADVERSATIVE_LANGUAGE``: although, but, despite, even
# though, except, however, nevertheless, nonetheless, notwithstanding,
# though, unless, whereas, while, yet. Any may take the conjunction ``ו``.
#
# Some are also ordinary words: ``אולם`` (hall), ``אבל`` (mourning),
# ``להוציא`` (to issue), ``זולת`` (another person), ``בעוד`` (in a stated
# time). Without punctuation the connective and the word read alike (``חסר
# מידע אולם סעיף 345ב קודד``, "however"), so each counts wherever it stands,
# except in a form or after a governor that fixes the ordinary word: a prefix
# other than ``ו`` (``האולם``, ``באבל``, ``הזולת``) is never the connective,
# ``להוציא`` after a modal (``יש להוציא``, ``כדי להוציא``) is the infinitive,
# and ``בעוד`` before a span of time (``בעוד שנה``, ``בעוד 30 יום``) is "in".
# A bare noun such as ``הגדרת אולם`` is refused, stricter than English.
_ADVERSATIVE_WORDS = (
    'ברם למרות חרף למעט מלבד ואילו ואולם מאידך מנגד אעפ״כ אעפ"כ אולם אבל זולת'
)
_INFINITIVE_GOVERNORS = (
    "יש צריך צריכה צריכים צריכות חייב חייבת חייבים חייבות רשאי רשאית רשאים "
    "רשאיות מוסמך מוסמכת מוסמכים מוסמכות יכול יכולה יכולים יכולות ניתן אפשר "
    "כדי בכדי מנת עליו עליה עליהם עליהן"
)
# Each governor is one word, one space before the infinitive; a lookbehind
# per governor keeps each one fixed-width. The lookahead runs them only where
# ``להוציא`` follows, so a scan does not pay for them at every word.
_GOVERNED_INFINITIVE = "(?=להוציא)" + "".join(
    rf"(?<!(?<![{_H}]){governor} )" for governor in _INFINITIVE_GOVERNORS.split()
)
_TIME_SPAN = (
    r"(?:(?:\d+|כמה|מספר)\s+)?"
    r"(?:שנה|שנים|שנתיים|חודש|חודשים|חודשיים|שבוע|שבועות|שבועיים|יום|ימים"
    r"|יומיים|שעה|שעות|שעתיים|דקה|דקות|זמן|מועד)"
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
    "הגם כי",
    "לעומת זאת",
    "עם כל זאת",
    "עם כל זה",
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
    ("אע״פ", "ש"),
    ('אע"פ', "ש"),
    ("אעפ״י", "ש"),
    ('אעפ"י', "ש"),
    # English lists "while"; these are its Hebrew forms.
    ("בשעה", "ש"),
    ("בעת", "ש"),
    ("בזמן", "ש"),
)
ADVERSATIVE_LANGUAGE = (
    # ``להוציא`` ("except"), unless a modal makes it "to issue".
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?{_GOVERNED_INFINITIVE}להוציא{_NOT_HEBREW_AFTER}|"
    # ``בעוד`` ("while"), unless a span of time follows ("in a year").
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?בעוד{_NOT_HEBREW_AFTER}"
    rf"(?!{_SEP}{_TIME_SPAN}{_NOT_HEBREW_AFTER})|"
    # ``אך`` ("but"), but not ``אך ורק`` ("only").
    rf"{_NOT_HEBREW_BEFORE}(?:ו)?אך{_NOT_HEBREW_AFTER}(?!{_SEP}ורק{_NOT_HEBREW_AFTER})|"
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
# The adversatives a reason is read for: ``_ADVERSATIVE_LANGUAGE`` in
# source_completeness, verbatim, the English and German words with the Hebrew
# ones, so a reason that reverses itself in another language (``חסר סעיף 2.
# However it is available.``) is refused as English refuses it.
# ``test_reason_adversatives_use_the_shared_vocabulary`` holds them equal.
_ANY_ADVERSATIVE = re.compile(
    r"\b(?:"
    r"although|but|despite|even\s+though|except|however|nevertheless|nonetheless|"
    r"notwithstanding|though|unless|whereas|while|yet|aber|jedoch|obwohl"
    r")\b|" + ADVERSATIVE_LANGUAGE,
    flags=re.IGNORECASE,
)
# English does not read the "yet" of "not yet" as an adversative
# (``_has_adversative_language``); blanking it keeps every offset.
_NOT_YET = re.compile(r"\bnot\s+yet\b", flags=re.IGNORECASE)

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
    r"\s*(?!(?:"
    + "|".join(_NAME_STOP_WORDS.split())
    + rf"){_NOT_HEBREW_AFTER})[{_QUOTES}]?[{_H}][{_H}{_GERESH_MARKS}{_MAQAF}{_QUOTES}-]*"
)
_INSTRUMENT_NAME = (
    # ``חוק יסוד: כבוד האדם וחירותו``, ``חוק־יסוד:חופש העיסוק``.
    rf"(?:\s*(?:[{_MAQAF}-]\s*)?יסוד\s*:)?"
    rf"(?:{_NAME_WORD}){{0,6}}"
    r"(?:\s*\([^()\n]{1,40}\))?"
)
# What may follow a name that was cut short: another name word past the
# six-word bound, a parenthetical too long or nested to read, a quoted or
# colon-led continuation, or a number. Such a name identifies no law with
# certainty.
_NAME_CONTINUES = re.compile(
    rf"(?:{_NAME_WORD})|\s*\(|\s*(?:[{_QUOTES}]\s*)?:\s*[{_H}]|\s*[{_MAQAF}-][{_H}]"
    # A number continues a title (``חוק התקציב לשנת הכספים 2025``), and so do
    # an edition (``[נוסח משולב]``) and an enactment date (``, התשנ״ה–1995``).
    r"|\s*\d|\s*\["
    rf"|\s*,\s*(?:ה?תש[{_H}{_GERESH_MARKS}]*|\d{{4}})"
)
SECTION_DEPENDENCY = re.compile(
    # A single section, as English's grammar reads ``section`` and not
    # ``sections``; a suffix it cannot read (``345b``) leaves it unread.
    rf"{_NOT_HEBREW_BEFORE}(?P<proclitics>[ובלמשכה]{{0,3}})סעיף\s+"
    rf"(?P<section>\d+{_SECTION_SUFFIX})(?![0-9A-Za-z{_H}{_GERESH_MARKS}])"
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
    current_kind: str = "חוק",
) -> Instrument:
    """Resolve ``לחוק זה``, ``לחוק האמור`` or ``לחוק הביטוח הלאומי``.

    ``זה`` names the current instrument only in its own kind: ``לפקודה זו``
    ("this ordinance") in a law is no instrument this reader knows.
    """

    first = words[0] if words else ""
    if first in _CURRENT_INSTRUMENT_NAMES:
        return _CURRENT_INSTRUMENT if kind == current_kind else _UNRESOLVED_INSTRUMENT
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
    text: str, current_kind: str = "חוק"
) -> tuple[tuple[re.Match[str], Instrument | None], ...]:
    """Return each section citation with the instrument it belongs to.

    ``None`` is a citation that names no instrument; in a source that is the
    current law, while a reason may leave the instrument to its context. An
    unqualified citation after another law named in its clause (``חוק הביטוח
    הלאומי: חסר סעיף 42``) may be that law's, so it is unresolved.
    """

    mentions = _explicit_instrument_mentions(text)
    preposed = _PreposedInstruments(text, current_kind)
    citations: list[tuple[re.Match[str], Instrument | None]] = []
    for match in SECTION_DEPENDENCY.finditer(text):
        kind = match.group("instrument_kind")
        if not kind:
            instrument = (
                _UNRESOLVED_INSTRUMENT if preposed.precede(match.start()) else None
            )
        elif _name_is_cut_short(text, match.end()):
            instrument = _UNRESOLVED_INSTRUMENT
        else:
            instrument = _named_instrument(
                _instrument_kind(kind),
                _instrument_words(match.group("instrument_name") or ""),
                mentions,
                match.start("instrument"),
                current_kind,
            )
        citations.append((match, instrument))
    return tuple(citations)


def _same_instrument(left: Instrument, right: Instrument) -> bool:
    """The same law, named in full: no prefix or abbreviation stands for another.

    ``חוק המקרקעין`` and ``חוק המקרקעין (חיזוק בתים משותפים מפני רעידות
    אדמה)`` are two statutes, so names compare whole.
    """

    return left == right and left != _UNRESOLVED_INSTRUMENT


def _dependency_is_external(
    match: re.Match[str], instrument: Instrument | None, current_section: str
) -> bool:
    if instrument is not None and instrument != _CURRENT_INSTRUMENT:
        return True
    section, _fragments = _dependency_identity(match)
    return section != _normalize_word(current_section)


# A clause ends at ``;``, a line break or a sentence period, not ``5.17``.
# Each scan finds its text's stops once; nothing caches them by text, since a
# lookup keyed by an equal text costs a full comparison per citation.
_CLAUSE_STOP = re.compile(r"[;\n]|\.(?!\d)")


def _clause_stops(text: str) -> list[int]:
    return [match.start() for match in _CLAUSE_STOP.finditer(text)]


# Where an instrument named earlier stops owning a later citation. A clause
# stop ends ownership, but a line break does so only where the next line opens
# an outline branch: any other line break wraps one reference (``חוק הביטוח
# הלאומי\nסעיף 14(ב)``), and the law named before it still owns the citation
# after it.
_OWNERSHIP_STOP = re.compile(rf";|\.(?!\d)|\n(?=[ \t]*\({OUTLINE_LABEL}\))")


# The linkers that make a cited section operative are English's: ``לפי``
# (under), ``על פי`` (according to) and ``בהתאם ל`` (in accordance with), as
# in ``_source_clause_links_dependency``. ``כהגדרתה ב`` ("as defined in") and
# ``הקבוע ב`` ("set in") are not, in English either.
#
# A negation, prohibition or exclusion before a linker in its clause negates
# the link (``אין לפעול בהתאם ל``, ``נאסר ...``, ``אי אפשר ...``). A
# main-clause negation governs to the end of its clause, relative clauses
# included. A negation inside a relative clause (``שאינו``, ``שלא``,
# ``דלא``) ends where a relative clause carrying a ``לפי``/``על פי`` link
# begins, the word just before the linker being its relative verb (``החלק
# שאינו עולה על הסכום שנקבע לפי סעיף 341``, section 14(ו1)); ``בהתאם ל``
# takes no reset, as English binds negation to ``in accordance with``. Any
# other negation stands, so ``מבוטח שלא עבד ישלם לפי`` is read as negating:
# stricter than English, which does not check ``under``. An English or German
# negator (``_OTHER_NEGATION`` or ``_OTHER_LINK_NEGATION``) is a main-clause
# negation.
_NEGATIONS = re.compile(
    rf"{_NOT_HEBREW_BEFORE}(?P<relative>ו?(?:(?:מ|כ|לכ|ב)?ש|ד))?(?:[ומה]{{0,2}})"
    r"(?:אין|לא|אינ(?:ו|ה|ם|ן|י|ך|כם|כן|נו|נה|ני)|אסור(?:ה|ים|ות)?|בל|לבל"
    r"|נאסר(?:ה|ו|ים|ות)?|איסור|מסרב(?:ת|ים|ות)?|בלא"
    # Prevented (``מנוע מלפעול``), not the noun "engine" (``מס על מנוע``).
    rf"|מנוע(?:ה|ים|ות)?(?={_NEGATION_GAP}+מ[{_H}])"
    # Refuses, will refuse, refused; unable (``נבצר מן המוסד``); fails,
    # will fail, failed, as English reads "refuses to", "unable to" and
    # "fails to".
    r"|יסרב(?:ו)?|תסרב(?:נה|ו)?|סי?רב(?:ה|ו)?|לסרב|נבצר(?:ה)?"
    r"|כושל(?:ת|ים|ות)?|נכשל(?:ה|ו|ים|ות)?|כשל(?:ה|ו)?|י?יכשל(?:ו)?|תי?כשל(?:נה|ו)?"
    # A prohibitive imperative (``אל תחשב``); ``אל`` before another word
    # starting with ``ת`` (``אל תושב``, "to a resident") is refused with it.
    rf"|אל(?={_NEGATION_GAP}+ת[{_H}])"
    # Excluded, prevented, rejects, denies (not ``דוחות``, "reports").
    r"|הוחרג(?:ה|ו)?|מוחרג(?:ת|ים|ות)?|יוחרג(?:ו)?|תוחרג(?:נה)?"
    r"|נמנע(?:ה|ו|ים|ות)?|יימנע(?:ו)?|תימנע(?:נה)?|דוח(?:ה|ים)"
    r"|שולל(?:ת|ים|ות)?|נשלל(?:ה|ו|ים|ות)?"
    r"|אוסר(?:ת|ים|ות)?|אסר(?:ה|ו)?|יאסור|תאסור|יאסרו|תאסו?רנה|ייאסר(?:ו)?|תיאסר(?:נה)?"
    r"|מכחיש(?:ה|ים|ות)?|הכחיש(?:ה|ו)?"
    # ``דוחות את הטענה`` ("reject the claim"), not ``דוחות`` ("reports").
    rf"|דוחות(?=\s+את{_NOT_HEBREW_AFTER})"
    # Except, other than: ``להוציא`` but not the infinitive after a modal
    # (``יש להוציא רישיון``, "a license must be issued"), and ``זולת`` but not
    # the noun ``הזולת`` ("another person").
    rf"|בלי|ללא|מבלי|למעט|מלבד|{_GOVERNED_INFINITIVE}להוציא|(?<!ה)זולת)"
    rf"{_NOT_HEBREW_AFTER}"
    # ``בלתי`` under any chain of prefix letters (``הבלתי``, ``לבלתי``); a
    # chain holding ``ש`` or ``ד`` opens a relative clause (``_is_relative``).
    rf"|{_NOT_HEBREW_BEFORE}(?P<bilti_prefix>[{_PREFIX_LETTERS}]*)בלתי"
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
_DISPLACEMENT = re.compile(
    r"(?:(?:מ)?בלי\s+לגרוע|על\s+אף|אין\s+ב[^.;\n]{0,80}כדי\s+לגרוע)"
    r"[^.;\n]{0,40}$"
)
_SOURCE_LINKER = re.compile(
    r"(?:"
    rf"{_NOT_HEBREW_BEFORE}(?:ו|ש|וש)?(?P<under>לפי|על{_SEP}פי|עפ[״\"]י)\s*"
    # ``בהתאם להוראות סעיף``, "in accordance with the provisions of section".
    rf"|{_NOT_HEBREW_BEFORE}(?:ו)?בהתאם\s+ל(?:הוראו?ת\s*)?"
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


def _is_relative(negation: re.Match[str]) -> bool:
    """Whether a ``_NEGATIONS`` match negates inside a relative clause.

    ``שלא``, ``דאינו`` and ``שהבלתי`` do; ``לא``, ``ולא`` and ``הבלתי`` are
    main-clause negations.
    """

    prefix = negation.group("bilti_prefix")
    return bool(negation.group("relative")) or bool(prefix and set(prefix) & {"ש", "ד"})


def _link_resets(text: str, negation: re.Match[str], linker: re.Match[str]) -> bool:
    """Whether a relative clause after a relative negation carries the link."""

    if not _is_relative(negation) or not linker.group("under"):
        return False
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

    def add(self, instrument: Instrument) -> None:
        if instrument == _CURRENT_INSTRUMENT:
            self.current = True
        elif instrument != _UNRESOLVED_INSTRUMENT:
            kind, words = instrument
            self.names.setdefault(kind, set()).add(words)

    def admits(self, instrument: Instrument) -> bool:
        """``_same_instrument`` against every entry, by one set lookup."""

        if instrument == _CURRENT_INSTRUMENT:
            return self.current
        if instrument == _UNRESOLVED_INSTRUMENT:
            return False
        kind, words = instrument
        return words in self.names.get(kind, ())


LinkedIndex = dict[tuple[str, tuple[str, ...]], _LinkedInstruments]


def _linked_source_instruments(
    source_scope_text: str,
    current_kind: str = "חוק",
    other_negations: Callable[[str], Iterable[re.Match[str]]] | None = None,
    other_link_negations: Callable[[str], Iterable[re.Match[str]]] | None = None,
) -> LinkedIndex:
    """Instruments of the operative citations, by section and fragment prefix.

    Each scope citation that a linker makes operative (``לפי סעיף 345ב(א)``)
    is entered under its section with every prefix of its fragments, so a
    reason citing ``345ב`` or ``345ב(א)`` is answered by one lookup. A scan
    builds this once and passes it down. ``other_negations`` and
    ``other_link_negations`` find the English and German negators,
    ``_OTHER_NEGATION`` and ``_OTHER_LINK_NEGATION`` without them.
    """

    index: LinkedIndex = {}
    stops = _clause_stops(source_scope_text)
    negations = list(_NEGATIONS.finditer(source_scope_text))
    negation_starts = [negation.start() for negation in negations]
    negation_ends = [negation.end() for negation in negations]
    # Main-clause negations among the first ``i``, so a range is counted in O(1).
    main_before = [0]
    for negation in negations:
        main_before.append(main_before[-1] + (not _is_relative(negation)))
    # Where the English and German negators start. Each is a main-clause
    # negation, read where it starts: one that starts in the linker's clause
    # before the linker negates it. The two vocabularies overlap, so only the
    # starts are kept, in order. They are read with scripts split, so a
    # Hebrew prefix glued to one (``וnot``, ``שohne``) hides it from neither.
    other_starts = _split_starts(
        source_scope_text,
        (
            other_negations or _OTHER_NEGATION.finditer,
            other_link_negations or _OTHER_LINK_NEGATION.finditer,
        ),
    )
    for cited, cited_instrument in _section_citations(source_scope_text, current_kind):
        clause_start = _clause_start(stops, cited.start())
        # The linkers and the displacement guard read at most ~140 characters
        # before the citation; the window keeps a long clause linear. The
        # negation index covers the whole clause.
        window_start = max(clause_start, cited.start() - _LINK_WINDOW)
        linker = _source_linker(
            source_scope_text, window_start, cited.end("proclitics")
        )
        if linker is None:
            continue
        other = bisect.bisect_left(other_starts, clause_start)
        if other < len(other_starts) and other_starts[other] < linker.start():
            continue
        first = bisect.bisect_left(negation_starts, clause_start)
        last = bisect.bisect_right(negation_ends, linker.start()) - 1
        if last >= first:
            if main_before[last + 1] - main_before[first]:
                continue
            if not _link_resets(source_scope_text, negations[last], linker):
                continue
        section, fragments = _dependency_identity(cited)
        for length in range(len(fragments) + 1):
            index.setdefault((section, fragments[:length]), _LinkedInstruments()).add(
                cited_instrument or _CURRENT_INSTRUMENT
            )
    return index


def _source_scope_links_dependency(
    linked_index: LinkedIndex,
    dependency: re.Match[str],
    instrument: Instrument | None,
) -> bool:
    """Bind a reason's dependency to an operative citation in the deferred scope.

    ``linked_index`` is the scope's ``_linked_source_instruments``. The scope
    must cite the same section (and at least the reason's fragments) of the
    same instrument. A reason that names no instrument takes the one the
    scope cites; one that names an instrument binds only a scope citation of
    that instrument, with ``לחוק האמור`` read as the law last named and an
    unqualified scope citation read as the current law.
    """

    linked = linked_index.get(_dependency_identity(dependency))
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


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char in _GERESH_MARKS


def _words_back(text: str, end: int, count: int) -> int:
    """Where the ``count`` words before ``end`` begin, separators however long.

    Words are letters, digits and numeral marks; anything else separates
    them, so ``חסר,חסר,...`` is many short words and no scan grows.
    """

    start = end
    for _ in range(count):
        while start and not _is_word_char(text[start - 1]):
            start -= 1
        if not start:
            break
        while start and _is_word_char(text[start - 1]):
            start -= 1
    return start


def _read_signals(
    text: str,
    stops: Iterable[int],
    other_negations: Callable[[str], Iterable[re.Match[str]]] | None = None,
) -> tuple[list[re.Match[str]], list[int]]:
    """Unnegated missing-dependency signals, and where the negations start.

    A signal is negated by a negation over the few words before it (``לא
    אמורה להיות חסרה``), however much whitespace sits between them, or by any
    negation earlier in its clause (``לא ניתן לומר שחסרה``, ``It is not true
    that חסרה``). The negations are those of ``_REASON_NEGATION`` and the
    English and German negators (``other_negations``, ``_OTHER_NEGATION``
    without it) outside every signal, read with scripts split
    (``_split_starts``), so ``וnot`` and ``שnicht`` negate as ``ו־not`` does.
    Each lookup is a bisection, so the scan stays linear in the reason's
    length.
    """

    stops = list(stops)
    every = list(_SIGNAL.finditer(text))
    signal_starts = [signal.start() for signal in every]
    negations = []
    for start in _split_starts(
        text, (_REASON_NEGATION.finditer, other_negations or _OTHER_NEGATION.finditer)
    ):
        holder = bisect.bisect_right(signal_starts, start) - 1
        if holder < 0 or start >= every[holder].end():
            negations.append(start)
    signals = []
    for signal in every:
        end = signal.start()
        while end and (text[end - 1].isspace() or text[end - 1] in _MAQAF + "-"):
            end -= 1
        start = _words_back(text, end, _NEGATION_REACH_WORDS)
        if _NEGATED_SIGNAL_PREFIX.search(text, start, end):
            continue
        stop = bisect.bisect_left(stops, signal.start())
        clause_start = stops[stop - 1] + 1 if stop else 0
        first = bisect.bisect_left(negations, clause_start)
        if first < len(negations) and negations[first] < signal.start():
            continue
        signals.append(signal)
    return signals, negations


def _reason_stops(
    reason: str,
    masked_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
) -> list[int]:
    """Clause stops, but not the periods inside the citations English masks.

    ``42 U.S.C. 9`` and ``R.S. 47:295`` hold periods that end no clause, as
    ``_reason_clause_bounds`` reads them; ``masked_citations`` finds them.
    """

    masked_spans = sorted(
        (match.start(), match.end())
        for match in (masked_citations(reason) if masked_citations else ())
    )
    stops = []
    span_index = 0
    for stop in _clause_stops(reason):
        while span_index < len(masked_spans) and masked_spans[span_index][1] <= stop:
            span_index += 1
        inside = (
            span_index < len(masked_spans)
            and masked_spans[span_index][0] <= stop < masked_spans[span_index][1]
        )
        if not inside:
            stops.append(stop)
    return stops


def _signals(text: str) -> list[re.Match[str]]:
    """Missing-dependency signals of ``text`` that no negation governs."""

    return _read_signals(text, _clause_stops(text))[0]


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
        citations: Sequence[tuple[re.Match[str], Instrument | None]] | None = None,
        other_negations: Callable[[str], Iterable[re.Match[str]]] | None = None,
    ) -> None:
        # Every adversative English reads, in any of its languages, with "not
        # yet" blanked as English blanks it, and scripts split, so a glued
        # ``וhowever`` reverses the reason as ``ו־however`` does.
        self.adversatives = _split_starts(
            reason,
            (
                lambda view: _ANY_ADVERSATIVE.finditer(
                    _NOT_YET.sub(lambda match: " " * len(match.group(0)), view)
                ),
            ),
        )
        # A citation starts at its ``סעיף`` word: its prefix letters may belong
        # to a signal (``תלוי בסעיף``), and the citation is still between that
        # signal and a later one. ``citations`` are the reason's
        # ``_section_citations``, when the caller has them.
        other_spans = sorted(
            (match.start(), match.end())
            for match in (other_citations or _OTHER_CITATION.finditer)(reason)
        )
        spans = sorted(
            [
                (match.end("proclitics"), match.end())
                for match, _ in (
                    _section_citations(reason) if citations is None else citations
                )
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
        self.stops = _reason_stops(reason, masked_citations)
        self.length = len(reason)
        self.signals, self.negations = _read_signals(
            reason, self.stops, other_negations
        )
        self.signal_ends = [signal.end() for signal in self.signals]

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

    def has_negation(self, start: int, end: int) -> bool:
        index = bisect.bisect_left(self.negations, start)
        return index < len(self.negations) and self.negations[index] < end

    def has_citation(self, start: int, end: int) -> bool:
        """Whether a citation lies wholly inside ``[start, end)``."""

        first = bisect.bisect_left(self.citation_starts, start)
        return (
            first < len(self.citation_starts) and self.earliest_end_from[first] <= end
        )


# English refuses a bridge of more than 240 characters between the signal and
# the citation. English translations of Hebrew legal prose run longer: 1.5
# times as long at the median of this module's test translations and 1.93
# times at most, so a Hebrew bridge equal to English's could exceed it in
# translation. A Hebrew bridge of 100 characters stays within 240 up to 2.4
# times.
_BRIDGE_LIMIT = 100


def reason_names_source_bound_dependency(
    reason: str,
    source_scope_text: str,
    *,
    corpus_citation_path: str,
    other_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
    masked_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
    other_negations: Callable[[str], Iterable[re.Match[str]]] | None = None,
    other_link_negations: Callable[[str], Iterable[re.Match[str]]] | None = None,
) -> bool:
    """Accept a Hebrew reason naming a missing external section the scope applies.

    The Hebrew form of the English external-dependency route, and no looser
    than it on the same text. The cited section must differ from the current
    one (or belong to another instrument), a missing-dependency signal must
    govern it within one clause with no negation or other citation in
    between, neither that clause nor a following citation-free clause may
    hold an adversative, and the deferred source scope must make that section
    of that instrument operative (``לפי סעיף 345ב``, ``כהגדרתה בסעיף 345ב``).
    Only the missing and without signals open it: the requires, until and
    cannot-compute routes rest on English vocabulary lists that Hebrew words
    do not translate into reliably, so a Hebrew reason cannot use them.
    The instrument comparison is stricter than the English prose route.
    ``other_citations`` is the English reader's citation finder (U.S. Code,
    R.S., ``§``, ``Absatz`` and canonical paths); without it the English
    precise-dependency grammar alone stands in. ``masked_citations`` are the
    citations whose periods the English clause reader masks.
    ``other_negations`` finds the English and German negators
    (``_NEGATION_LANGUAGE``), which negate a Hebrew signal or link as a Hebrew
    negation does; without it ``_OTHER_NEGATION``, its verbatim copy, does.
    ``other_link_negations`` finds the English negations that negate an ``in
    accordance with`` link (``_SOURCE_LINK_NEGATION_LANGUAGE``), which also
    negate a Hebrew link; without it ``_OTHER_LINK_NEGATION`` does.

    Reason and scope are read as one folded copy each (``fold_marks``), so
    points and direction marks hide no negation, adversative, signal or
    citation, and every offset compared is an offset into that copy. Each
    text is indexed once; no lookup inside the citation loop rereads one.
    """

    current_section = native_section_label(corpus_citation_path)
    if current_section is None or not source_scope_text.strip():
        return False
    reason = fold_marks(reason)
    source_scope_text = fold_marks(source_scope_text)
    current_kind = current_instrument_kind(corpus_citation_path)
    citations = _section_citations(reason, current_kind)
    index = _ReasonIndex(
        reason, other_citations, masked_citations, citations, other_negations
    )
    stops = index.stops
    linked_index: LinkedIndex | None = None
    for dependency, instrument in citations:
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
        # The signal is read up to the citation's own prefix letters
        # (``שבסעיף``), which belong to the citation.
        reference_start = dependency.end("proclitics")
        signal = index.last_signal(clause_start, reference_start)
        if signal is None:
            continue
        # Offsets are compared before anything is sliced, so many citations in
        # one long clause stay linear. A negation between the signal and the
        # citation leaves the missing claim unread.
        if (
            reference_start - signal.end() > _BRIDGE_LIMIT
            or index.has_citation(signal.end(), reference_start)
            or index.has_negation(signal.end(), reference_start)
        ):
            continue
        if not (signal.group("missing") or signal.group("without")):
            continue
        # As in English, an earlier citation in the same clause leaves the
        # missing claim ambiguous between the two.
        if index.has_citation(clause_start, signal.start()):
            continue
        if linked_index is None:
            linked_index = _linked_source_instruments(
                source_scope_text,
                current_kind,
                other_negations,
                other_link_negations,
            )
        if _source_scope_links_dependency(linked_index, dependency, instrument):
            return True
    return False


# --- Hebrew negations in English text ---------------------------------------
#
# The Hebrew reader takes English and German negators before a Hebrew signal or
# link (``not חסר סעיף 2``). The converse holds too: a Hebrew negation before
# an English signal or link (``לא missing section 2``, ``(a) לא computed in
# accordance with section 2``) is read as its translation's negation, by the
# shared checker's English routes (#1882). Text with no Hebrew script reads
# nothing here, so no English or German verdict changes.


def english_reading_view(
    reason: str,
    masked_citations: Callable[[str], Iterable[re.Match[str]]] | None = None,
) -> str | None:
    """The copy of a mixed-language reason the English routes must also accept.

    ``None`` when the reason has no Hebrew script. Otherwise it is
    ``cross_language_text(reason)`` with every clause blanked from each Hebrew
    negation (``_REASON_NEGATION``, outside the Hebrew signals that hold one,
    such as ``טרם קודד``) to the clause's end. That is the Hebrew reader's
    rule: a negation governs what follows it in its clause, before a signal or
    between a signal and its citation. So ``לא missing section 2`` and
    ``missing לא section 2`` cite nothing in this copy, and the English
    routes, which read no negation before ``missing``, refuse them. A
    citation before the negation (``missing section 2, שלא קודד``) and a
    clause of its own stay. The shared checker accepts a reason only if it
    accepts this copy as well, which refuses where the copy alone would
    refuse and admits nothing the reason alone would not. ``masked_citations``
    are the citations whose periods end no clause.
    """

    text = cross_language_text(reason)
    if text is None:
        return None
    signals = [(signal.start(), signal.end()) for signal in _SIGNAL.finditer(text)]
    signal_starts = [start for start, _ in signals]
    negations = []
    for negation in _REASON_NEGATION.finditer(text):
        holder = bisect.bisect_right(signal_starts, negation.start()) - 1
        if holder < 0 or negation.start() >= signals[holder][1]:
            negations.append(negation.start())
    if not negations:
        return text if text != reason else None
    stops = _reason_stops(text, masked_citations)
    parts = []
    kept = 0
    for start in negations:
        if start < kept:
            continue
        stop = bisect.bisect_left(stops, start)
        end = stops[stop] if stop < len(stops) else len(text)
        parts.append(text[kept:start])
        parts.append(" " * (end - start))
        kept = end
    parts.append(text[kept:])
    return "".join(parts)


def link_negations(text: str) -> Iterable[re.Match[str]]:
    """The Hebrew negations, prohibitions and exclusions of a source link.

    ``_NEGATIONS``, for the shared checker's English link guard, which reads
    them in ``cross_language_text`` as it reads ``not`` and ``except``.
    """

    return _NEGATIONS.finditer(text)


def ends_with_negation(text: str) -> bool:
    """Whether a mixed-language text ends with a Hebrew link negation.

    The German reader refuses ``nicht abweichend von``; a Hebrew negation in
    ``nicht``'s place (``לא abweichend von``) is read the same way.
    """

    view = cross_language_text(text)
    if view is None:
        return False
    end = len(view.rstrip())
    return any(negation.end() == end for negation in _NEGATIONS.finditer(view))


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
    # A rate starts where its digits start: a match from inside a run of
    # digits implies one from the run's first digit, so the lookbehind changes
    # no verdict and keeps a long run of digits from being retried at every
    # suffix.
    rf"(?<!\d)\d+(?:[.,]\d+)?\s*%\s*(?:ו)?מ(?:ן\s)?(?=[{_H}])"
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
    return _EXPLICIT_COMPUTATION.search(fold_marks(text)) is not None
