"""Shared Hebrew prefix joints and bounded regex fragments.

Unicode's dash category is the closed class of printed prefix joints, plus
U+2212 MINUS SIGN. A joint keeps its punctuation; only horizontal spaces
around a joint between a standalone prefix chain and a Hebrew word fold.

A run of two or three prefix letters before a dash is a prefix chain when
grammar can stack those proclitics and the run is not itself a word, or when
the run is ambiguous (AMBIGUOUS_CHAIN_RUNS). Every other run reads as a word,
and the dash after it is punctuation. No dictionary decides this.
"""

import itertools
import re
import sys
import unicodedata
from collections.abc import Iterator

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
HORIZONTAL_SPACES = frozenset(
    "\t \u00a0\u1680" + "".join(map(chr, range(0x2000, 0x200B))) + "\u202f\u205f\u3000"
)
_HORIZONTAL_SPACE = "[" + re.escape("".join(sorted(HORIZONTAL_SPACES))) + "]"
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
# review's ordinary-name control. Both compact and spaced dashes after these
# complete words remain punctuation in every prefix reader.
STATUTORY_PREFIX_WHOLE_WORDS = frozenset(
    "בדו בה בו בכל בל בלב בשל דו דוד הכל הלב השב השד ובו ובה וכל ולד ושב ושד ושל ושש "
    "כבד כד כה ככל כל כלל כמה כמו כשל לב לבד לה לו לכל למד לשד מד מדד מה מהו "
    "מכל ממש משל משש משה שב שבה שבו שד שהה שוב שכל של שלב שלה שלו שמה שמו שש ששב ששה".split()
)
# Proclitics stack in one order: the conjunction ו; a subordinator (ש, כש, מש,
# לכש, and the legal-register בש and Aramaic ד, the subordinators of
# hebrew_negators.ORDINARY_PREFIX_CHAINS); one preposition (ב, כ, ל, מ); then
# the article ה. Unpointed spelling drops the article after ב, כ and ל
# (בבית, not בהבית), so ה follows only ו, ש, כש, מש or מ. Prepositions follow
# only ש, כש and מש, and ד takes neither a preposition nor the article.
# Gesenius–Kautzsch–Cowley (1910), § 35n: ב, כ and ל elide the article's ה,
# and the conjunction ו does not:
# https://en.wikisource.org/wiki/Gesenius%27_Hebrew_Grammar/35._The_Article
_CONJUNCTIONS = ("", "ו")
_SUBORDINATORS = ("", "ש", "כש", "מש", "לכש", "בש", "ד")
_PREPOSITIONS = ("", "ב", "כ", "ל", "מ")


def _proclitic_sequences() -> Iterator[str]:
    for conjunction in _CONJUNCTIONS:
        for subordinator in _SUBORDINATORS:
            prepositions = (
                _PREPOSITIONS if subordinator in ("", "ש", "כש", "מש") else ("",)
            )
            for preposition in prepositions:
                article = preposition in ("", "מ") and subordinator not in ("בש", "ד")
                for ending in ("", "ה") if article else ("",):
                    sequence = conjunction + subordinator + preposition + ending
                    if 2 <= len(sequence) <= 3:
                        yield sequence


# The finite class of two/three-letter proclitic sequences (37 runs).
GRAMMATICAL_PREFIX_SEQUENCES = frozenset(_proclitic_sequences())
# Sequences in that class that are also complete words, absent from the
# statute vocabulary above. Glosses follow ordinary Hebrew usage.
GRAMMAR_PREFIX_WHOLE_WORDS = frozenset(
    (
        "מש",  # moved away, budged: לא מש ממקומו, "did not budge"
        "ומש",  # and moved away
        "שה",  # lamb, sheep
        "ושה",  # and a lamb
        "כשה",  # like a lamb
        "כשב",  # as one who returns (כ + שב)
        "ומה",  # and what
        "משב",  # gust: משב רוח, "a gust of wind"
    )
)
PREFIX_WHOLE_WORDS = STATUTORY_PREFIX_WHOLE_WORDS | GRAMMAR_PREFIX_WHOLE_WORDS
# The runs a dash joins to the next word. Any two/three-letter run outside
# the proclitic class, a word or not, cannot stack as proclitics: a dash after
# it is punctuation and the next word stays visible. Two kinds of run are
# ambiguous instead, and a dash after either joins as before. A word never ends
# in the non-final forms כ and מ (a final letter is ך or ם), so a run ending in
# either is never a complete word. And וו is the word "hook", but also the
# conjunction ו before a word that begins with ו (וואילו, "and whereas").
_NON_FINAL_FORMS = "כמ"
GRAMMATICAL_CHAIN_RUNS = GRAMMATICAL_PREFIX_SEQUENCES - PREFIX_WHOLE_WORDS
AMBIGUOUS_CHAIN_RUNS = (
    frozenset(
        {
            "".join(letters) + final
            for size in (1, 2)
            for letters in itertools.product(HEBREW_PREFIX_LETTERS, repeat=size)
            for final in _NON_FINAL_FORMS
        }
        | {"וו"}
    )
    - GRAMMATICAL_CHAIN_RUNS
)
PREFIX_CHAIN_RUNS = GRAMMATICAL_CHAIN_RUNS | AMBIGUOUS_CHAIN_RUNS


def joins_at_dash(run: str) -> bool:
    """Whether a dash after this run of prefix letters is a prefix joint."""

    return len(run) == 1 or run in PREFIX_CHAIN_RUNS


def _alternation(runs: frozenset[str] | set[str]) -> str:
    return "|".join(map(re.escape, sorted(runs)))


_TOKEN_BOUNDARY = r"(?<![א-תA-Za-z0-9_])"
# A letter of a script other than Hebrew. Where one meets a Hebrew letter,
# the cross-language copy (hebrew_source_text._split_scripts) inserts a space.
OTHER_SCRIPT_LETTER = r"[^\W\d_֐-׿]"
# English ``\b`` reads a word only between a word character of any script and
# none. The readers that admit a deferral (signals, links, citations) read a
# Hebrew word the same way: glued to a letter or digit of another script
# (``Яחסר``, ``חסרX``, ``٣סעיף``), it is part of one token, as its English
# (``ЯMissing``, ``MissingX``) is. _TOKEN_BOUNDARY above names only the
# Hebrew, Latin and ASCII word characters, so readers that refuse keep it:
# they read a Hebrew negator after Cyrillic (``Яלא``), the stricter reading.
WORD_START = r"(?<!\w)"
WORD_END = r"(?!\w)"


def _joining_run(length: int, chains: frozenset[str]) -> str:
    """The runs of this length in ``chains``, as one fixed-width alternative."""

    runs = {run for run in chains if len(run) == length}
    return f"(?:{_alternation(runs)})" if runs else ""


# A printed joint inside a prefix spelling stands only after a run that joins.
# The run reaches back to the token boundary, so each letter of ו־ש־ joins,
# while no reader crosses the punctuation after a word run (של, כלב) or after
# four or more letters. The lookahead keeps this off every unjointed letter.
_INNER_JOINT = (
    rf"(?:(?={PREFIX_JOINT})(?:(?<={_TOKEN_BOUNDARY}[{HEBREW_PREFIX_LETTERS}])"
    + "".join(
        rf"|(?<={_TOKEN_BOUNDARY}{_joining_run(length, PREFIX_CHAIN_RUNS)})"
        for length in (2, 3)
    )
    + rf"){PREFIX_JOINT})?"
)
# Every remaining prefix-only corpus token is accounted for: דמ is the
# gazette abbreviation דמ״י; הדו/ודו/לדו are pieces of דו״ח (report); וב/וה
# are date prefixes (וב־1/וה־1); וד is an outline label; וכד is וכד׳ (etc.);
# כב/כו/לד/מב are numeral labels. None is an ordinary complete word.
PREFIX_CORPUS_ABBREVIATIONS = frozenset(
    "דמ הדו וב וד ודו וה וכד כב כו לד לדו מב".split()
)


# Reject a restart only after an actual standalone prefix chain. A full word
# ending in a prefix letter (טל, חשמל), a whole prefix-only word (כל), or any
# run outside the proclitic class (כלב, כלה) followed by dash punctuation must
# leave the next standalone negator visible. Each lookbehind consumes exactly
# length+1 characters; its alternation is fixed width. No unbounded backscan
# or variable-width lookbehind is needed.
def _not_after_jointed_prefix(
    *,
    article: bool = False,
    chains: frozenset[str] = PREFIX_CHAIN_RUNS,
    token: str = _TOKEN_BOUNDARY,
) -> str:
    assertions = []
    for length in range(1, 4):
        if length == 1:
            chain = "ה" if article else f"[{HEBREW_PREFIX_LETTERS}]"
        else:
            runs = frozenset(run for run in chains if not article or run.endswith("ה"))
            chain = _joining_run(length, runs)
            if not chain:
                continue
        assertions.append(rf"(?<!{token}{chain}{PREFIX_JOINT})")
    # All assertions end in a joint; without a preceding joint they are
    # trivially true. Avoid their lexical work at ordinary boundaries.
    return rf"(?:(?<!{PREFIX_JOINT})|" + "".join(assertions) + ")"


PREFIX_BOUNDARY = _TOKEN_BOUNDARY + _not_after_jointed_prefix()
# PREFIX_BOUNDARY for the readers that admit a deferral (signals, links,
# citations), with token starts read as English ``\b`` reads them
# (WORD_START): a run glued to a letter of any script is part of a word, so a
# dash after ``Яו`` is punctuation, as after ``Xו``.
ADMIT_BOUNDARY = WORD_START + _not_after_jointed_prefix(token=WORD_START)
# An ambiguous run (ככ, דמ, וו) has two readings of its dash, and each reader
# takes the stricter one: readers that admit a deferral (signals, links,
# citations) take the joined reading above, while readers that refuse one
# (negators, exclusions, adversatives) restart after any run that is not a
# grammatical proclitic stack, and still read the joined spelling too.
REFUSAL_BOUNDARY = _TOKEN_BOUNDARY + _not_after_jointed_prefix(
    chains=GRAMMATICAL_CHAIN_RUNS
)
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
        if not joins_at_dash(prefix):
            continue
        for gap in _HORIZONTAL_RUN.finditer(text, chain.start(), chain.end()):
            yield gap.span()


def compact_prefix_joints(text: str) -> str:
    """Keep printed joints while removing their grammatical horizontal gaps."""

    def compact(chain: re.Match[str]) -> str:
        spelling = chain.group()
        prefix = re.split(_JOINT_GAP, spelling)[0]
        return _HORIZONTAL_RUN.sub("", spelling) if joins_at_dash(prefix) else spelling

    return _SPACED_PREFIX_CHAIN.sub(compact, text)


def prefix_word(prefix: str) -> str:
    """A literal prefix chain, with an optional printed joint after each letter."""

    return "".join(re.escape(letter) + _INNER_JOINT for letter in prefix)


def prefix_chain(letters: str, maximum: int = 3) -> str:
    """Zero to three prefix letters, retaining joints without unbounded state."""

    if maximum == 0:
        return ""
    return rf"(?:(?:[{re.escape(letters)}]{_INNER_JOINT}){{1,{maximum}}})?"
