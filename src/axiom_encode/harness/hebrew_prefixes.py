"""Shared Hebrew prefix joints and bounded regex fragments.

Unicode's dash category is the closed class of printed prefix joints, plus
U+2212 MINUS SIGN. A joint keeps its punctuation; only horizontal spaces
around a joint between a standalone prefix chain and a Hebrew word fold.
"""

import re
import sys
import unicodedata
from collections.abc import Iterator

from axiom_encode.harness._hebrew_prefix_words import HSPELL_PREFIX_WORDS

# Pd includes the maqaf, small hyphen-minus and fullwidth hyphen-minus. The
# Unicode database in the interpreter defines the finite class; its complete
# membership is pinned by the family test, rather than selected glyphs.
PREFIX_JOINT_CHARS = "".join(
    chr(code)
    for code in range(sys.maxunicode + 1)
    if unicodedata.category(chr(code)) == "Pd" or code == 0x2212
)
PREFIX_JOINT = "[" + re.escape(PREFIX_JOINT_CHARS) + "]"
HEBREW_PREFIX_LETTERS = "והבכלמשד"
# A reader cannot restart at a suffix just after a joint. Otherwise ``מ־לא``
# would read as the standalone ``לא`` although ``מלא`` means "full". Spaced
# joints first compact through prefix_joint_gaps, so this remains fixed width.
# Horizontal Unicode White_Space, excluding all vertical separators. Python's
# \s additionally includes U+001C–U+001F; those controls are folded elsewhere.
_HORIZONTAL_SPACE = "[\t \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
_JOINT_GAP = rf"{_HORIZONTAL_SPACE}*{PREFIX_JOINT}{_HORIZONTAL_SPACE}*"
_SPACED_PREFIX_CHAIN = re.compile(
    rf"(?<![א-תA-Za-z0-9_])"
    rf"(?:[{HEBREW_PREFIX_LETTERS}](?:{_JOINT_GAP})?){{0,2}}"
    rf"[{HEBREW_PREFIX_LETTERS}]{_JOINT_GAP}(?=[א-ת])"
)
_HORIZONTAL_RUN = re.compile(_HORIZONTAL_SPACE + "+")
# Complete two/three-letter words in the pinned Israeli-statute vocabulary
# (tests/fixtures/hebrew_negator_corpus.json). They happen to use only prefix
# letters, but a following dash is word punctuation. Include משה (Moses), the
# review's ordinary-name control. Compact legacy prefix spellings still read
# exactly as before; these exclusions govern the new horizontal-gap fold.
STATUTORY_PREFIX_WHOLE_WORDS = frozenset(
    "בדו בה בו בכל בל בלב בשל דו דוד הכל הלב השב השד ובו ובה וכל ולד ושב ושד ושל ושש "
    "כבד כד כה ככל כל כלל כמה כמו כשל לב לבד לה לו לכל למד לשד מד מדד מה מהו "
    "מכל ממש משל משש משה שב שבה שבו שד שהה שוב שכל של שלב שלה שלו שמה שמו שש ששב ששה".split()
)
# Hspell closes the finite 64+512 two/three-letter class, including inflections
# and ordinary prefixed words missing from the statute snapshot (כלב, בד, etc.).
# Keep the independently audited statutory words as an explicit overlay.
PREFIX_WHOLE_WORDS = HSPELL_PREFIX_WORDS | STATUTORY_PREFIX_WHOLE_WORDS
# Most reads are not at a joint. Require its bounded syntactic shape before
# trying the finite lexical alternation, so ordinary prose keeps its fast path.
PREFIX_NOT_WHOLE_WORD_JOINT = (
    rf"(?!(?=[{HEBREW_PREFIX_LETTERS}]{{2,3}}{PREFIX_JOINT})"
    + "(?:"
    + "|".join(map(re.escape, sorted(PREFIX_WHOLE_WORDS)))
    + ")"
    + PREFIX_JOINT
    + ")"
)
# Every remaining prefix-only corpus token is accounted for: דמ is the
# gazette abbreviation דמ״י; הדו/ודו/לדו are pieces of דו״ח (report); וב/וה
# are date prefixes (וב־1/וה־1); וד is an outline label; וכד is וכד׳ (etc.);
# כב/כו/לד/מב are numeral labels. None is an ordinary complete word.
PREFIX_CORPUS_ABBREVIATIONS = frozenset(
    "דמ הדו וב וד ודו וה וכד כב כו לד לדו מב".split()
)

# Reject a restart only after an actual standalone prefix chain. A full word
# ending in a prefix letter (טל, חשמל), or a whole prefix-only word (כל),
# followed by dash punctuation must leave the next standalone negator visible.
# Each lookbehind consumes exactly length+1 characters; its nested assertions
# are zero-width. No unbounded backscan or variable-width lookbehind is needed.
_TOKEN_BOUNDARY = r"(?<![א-תA-Za-z0-9_])"


def _not_after_jointed_prefix(*, article: bool = False) -> str:
    assertions = []
    for length in range(1, 4):
        words = "|".join(
            re.escape(word)
            for word in sorted(PREFIX_WHOLE_WORDS)
            if len(word) == length
        )
        not_word = rf"(?!(?:{words}){PREFIX_JOINT})" if words else ""
        chain = (
            rf"[{HEBREW_PREFIX_LETTERS}]{{{length - 1}}}ה"
            if article
            else rf"[{HEBREW_PREFIX_LETTERS}]{{{length}}}"
        )
        assertions.append(rf"(?<!{_TOKEN_BOUNDARY}{not_word}{chain}{PREFIX_JOINT})")
    # All three assertions end in a joint; without a preceding joint they
    # are trivially true. Avoid their lexical work at ordinary boundaries.
    return rf"(?:(?<!{PREFIX_JOINT})|" + "".join(assertions) + ")"


PREFIX_BOUNDARY = _TOKEN_BOUNDARY + _not_after_jointed_prefix()
# A definite article changes חסרה/חסרות into the deduction noun, and הזולת
# into "another person". Only an actual article prefix can suppress a reader;
# a complete preceding word ending in ה followed by punctuation cannot.
NOT_ARTICLE_PREFIX = r"(?<!ה)" + _not_after_jointed_prefix(article=True)


def prefix_joint_gaps(text: str) -> Iterator[tuple[int, int]]:
    """Horizontal spans inside validated one-to-three-letter prefix joints.

    Full words, numeric ranges, number-word dashes and line breaks remain
    punctuation. Each match is bounded to three prefix letters; long runs of
    prefix letters therefore use constant regex memory.
    """

    for chain in _SPACED_PREFIX_CHAIN.finditer(text):
        prefix = re.split(_JOINT_GAP, chain.group())[0]
        if prefix in PREFIX_WHOLE_WORDS:
            continue
        for gap in _HORIZONTAL_RUN.finditer(text, chain.start(), chain.end()):
            yield gap.span()


def compact_prefix_joints(text: str) -> str:
    """Keep printed joints while removing their grammatical horizontal gaps."""

    def compact(chain: re.Match[str]) -> str:
        spelling = chain.group()
        prefix = re.split(_JOINT_GAP, spelling)[0]
        return (
            spelling
            if prefix in PREFIX_WHOLE_WORDS
            else _HORIZONTAL_RUN.sub("", spelling)
        )

    return _SPACED_PREFIX_CHAIN.sub(compact, text)


def prefix_word(prefix: str) -> str:
    """A literal prefix chain, with an optional printed joint after each letter."""

    spelling = "".join(re.escape(letter) + PREFIX_JOINT + "?" for letter in prefix)
    return (
        PREFIX_NOT_WHOLE_WORD_JOINT + spelling
        if prefix in PREFIX_WHOLE_WORDS
        else spelling
    )


def prefix_chain(letters: str, maximum: int = 3) -> str:
    """Zero to three prefix letters, retaining joints without unbounded state."""

    if maximum == 0:
        return ""
    # Keep the zero-prefix alternative outside the guard: a genuine negative
    # core such as כשל may itself be a whole word before dash punctuation.
    return (
        rf"(?:{PREFIX_NOT_WHOLE_WORD_JOINT}"
        rf"(?:[{re.escape(letters)}]{PREFIX_JOINT}?){{1,{maximum}}})?"
    )
