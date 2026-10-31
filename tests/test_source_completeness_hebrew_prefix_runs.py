"""Round 17: every short prefix-letter run has one reading before a dash.

A run of two or three of the eight prefix letters joins the next word at a
dash only when grammar stacks it as proclitics and it is not itself a word, or
when it ends in a non-final letter, which no word does. Every other run, a
word or not, is followed by punctuation. No dictionary decides this: the
statute vocabulary and a short glossed list only mark the proclitic sequences
that are also words.
"""

from __future__ import annotations

import itertools

import pytest

from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness.hebrew_negators import NEGATOR_FAMILIES
from axiom_encode.harness.hebrew_prefixes import (
    AMBIGUOUS_CHAIN_RUNS,
    GRAMMAR_PREFIX_WHOLE_WORDS,
    GRAMMATICAL_CHAIN_RUNS,
    GRAMMATICAL_PREFIX_SEQUENCES,
    HEBREW_PREFIX_LETTERS,
    PREFIX_CHAIN_RUNS,
    PREFIX_WHOLE_WORDS,
    STATUTORY_PREFIX_WHOLE_WORDS,
    joins_at_dash,
)
from tests import test_source_completeness_hebrew as pinned
from tests.test_source_completeness_hebrew_prefix_joints import (
    _assert_source_issue_parity,
    _issues,
)

RUNS = tuple(
    "".join(letters)
    for size in range(1, 4)
    for letters in itertools.product(HEBREW_PREFIX_LETTERS, repeat=size)
)
WORD_RUNS = tuple(run for run in RUNS if not joins_at_dash(run))
JOINING_RUNS = tuple(run for run in RUNS if len(run) > 1 and joins_at_dash(run))
# Runs ending in a non-final letter that grammar cannot stack (ככ, דמ), and וו.
AMBIGUOUS_RUNS = tuple(run for run in JOINING_RUNS if run not in GRAMMATICAL_CHAIN_RUNS)
JOINTS = ("־", " - ")


def test_proclitic_class_is_the_written_grammar():
    """Conjunction, subordinator, preposition, article: the 37 short stacks."""

    assert GRAMMATICAL_PREFIX_SEQUENCES == frozenset(
        "בש וב וד וה וכ ול ומ וש כש מה מש שב שה שכ של שמ "
        "ובש וכש ומה ומש ושב ושה ושכ ושל ושמ כשב כשה כשכ כשל כשמ לכש "
        "משב משה משכ משל משמ שמה".split()
    )
    # The article never follows ב, כ or ל in unpointed spelling.
    assert not {"בה", "כה", "לה"} & GRAMMATICAL_PREFIX_SEQUENCES


def test_proclitic_words_come_from_the_statutes_or_the_glossed_list():
    assert PREFIX_WHOLE_WORDS == STATUTORY_PREFIX_WHOLE_WORDS | (
        GRAMMAR_PREFIX_WHOLE_WORDS
    )
    assert GRAMMAR_PREFIX_WHOLE_WORDS <= GRAMMATICAL_PREFIX_SEQUENCES
    assert not GRAMMAR_PREFIX_WHOLE_WORDS & STATUTORY_PREFIX_WHOLE_WORDS
    assert GRAMMATICAL_PREFIX_SEQUENCES & PREFIX_WHOLE_WORDS == frozenset(
        "מה שב של ושב ושל כשל משל משה שמה "  # statute vocabulary
        "מש ומש שה ושה כשה ומה משב כשב".split()  # glossed list
    )
    assert GRAMMATICAL_CHAIN_RUNS == frozenset(
        "בש וב וד וה וכ ול ומ וש כש שכ שמ ובש וכש ושכ ושמ כשכ כשמ לכש משכ משמ".split()
    )


def test_every_prefix_letter_run_has_exactly_one_reading():
    assert len(RUNS) == 584
    for run in RUNS:
        expected = (
            len(run) == 1
            or run in GRAMMATICAL_CHAIN_RUNS
            or run.endswith(("כ", "מ"))
            or run == "וו"
        )
        assert joins_at_dash(run) == expected, run
        assert (run in PREFIX_CHAIN_RUNS) == (len(run) > 1 and expected), run
    assert len(WORD_RUNS) == 421
    assert len(JOINING_RUNS) == 155
    assert len(AMBIGUOUS_RUNS) == 135
    assert set(AMBIGUOUS_RUNS) == AMBIGUOUS_CHAIN_RUNS
    assert not set(WORD_RUNS) & PREFIX_CHAIN_RUNS
    assert all(run in WORD_RUNS for run in PREFIX_WHOLE_WORDS)


REASON_BODIES = (
    "לא חסר סעיף 2",
    "חסר סעיף 2",
    "חסרה הוראה בסעיף 2",
    "זולת חסר סעיף 2",
    "מעט חסר סעיף 2",
)
SOURCE_FRAMES = (
    "(א) מס על {run}{joint}לא מחושב בהתאם לסעיף 2.",
    "(א) ישולם {run}{joint}לפי סעיף 2.",
    "(א) מס על {run}{joint}זולת תשלום לפי סעיף 2.",
    "(א) מס על {run}{joint}מעט חישוב בהתאם לסעיף 2.",
    "(א) מס על {run}{joint}לבד חישוב בהתאם לסעיף 2.",
)


# Every Unicode dash family is exhaustive in the joints module; here the maqaf
# keeps issue strings comparable (no diagnostic contains one) and the other
# dashes compare verdicts.
VERDICT_JOINTS = ("-", " - ", "\u2013", " \u05be ")
NUMERIC_RECALL = "[complete-source-unit:numeric-recall]"


def _maqaf_as_space(issues: tuple[str, ...]) -> tuple[str, ...]:
    # Number words (שש, "six") are the numeric reader's business: a dash after
    # one is not a space to it. The prefix readers' diagnostics must agree.
    return tuple(
        " ".join(issue.replace("\u05be", " ").split())
        for issue in issues
        if not issue.startswith(NUMERIC_RECALL)
    )


@pytest.mark.parametrize("body", REASON_BODIES)
def test_every_word_run_is_followed_by_punctuation_in_a_reason(body):
    """All 421 word runs read as a word, a space and the next word."""

    for run in WORD_RUNS:
        spaced = _issues(pinned.REVIEW_SOURCE, run + " " + body)
        joined = _issues(pinned.REVIEW_SOURCE, run + "\u05be" + body)
        assert _maqaf_as_space(joined) == _maqaf_as_space(spaced), (run, body)
        for joint in VERDICT_JOINTS:
            assert bool(_issues(pinned.REVIEW_SOURCE, run + joint + body)) == bool(
                spaced
            ), (run, joint, body)


@pytest.mark.parametrize("frame", SOURCE_FRAMES)
def test_every_word_run_is_followed_by_punctuation_in_a_source(frame):
    for run in WORD_RUNS:
        spaced = frame.format(run=run, joint=" ")
        expected = pinned._accepted("a", "חסר סעיף 2", spaced)
        joined = frame.format(run=run, joint="\u05be")
        # The same width keeps every locator; only its excerpt differs.
        assert _maqaf_as_space(_issues(joined, "חסר סעיף 2")) == _maqaf_as_space(
            _issues(spaced, "חסר סעיף 2")
        ), (run, frame)
        for joint in VERDICT_JOINTS:
            actual = pinned._accepted(
                "a", "חסר סעיף 2", frame.format(run=run, joint=joint)
            )
            assert actual == expected, (run, joint, frame)


# Bodies that admit a deferral, or that read the joined spelling as a
# prefixed lexeme (ל + מעט, מ + לבד).
JOINED_REASON_BODIES = ("חסר סעיף 2", "חסרה הוראה בסעיף 2", "מעט חסר סעיף 2")
JOINED_SOURCE_FRAMES = SOURCE_FRAMES[1:]


@pytest.mark.parametrize("joint", JOINTS)
def test_every_joining_run_reads_like_its_glued_spelling(joint):
    """All 155 joining runs before affirmative and lexical bodies."""

    for run in JOINING_RUNS:
        for body in JOINED_REASON_BODIES:
            assert _issues(pinned.REVIEW_SOURCE, run + joint + body) == _issues(
                pinned.REVIEW_SOURCE, run + body
            ), (run, joint, body)
        for frame in JOINED_SOURCE_FRAMES:
            glued = frame.format(run=run, joint="")
            split = glued.index(run) + len(run)
            _assert_source_issue_parity(
                glued,
                glued[:split] + joint + glued[split:],
                split,
                "חסר סעיף 2",
                _issues(glued, "חסר סעיף 2"),
            )


@pytest.mark.parametrize("joint", JOINTS)
def test_grammatical_chains_also_join_before_negators(joint):
    """ומ־לא is ומלא ("and full"); וש־לא is ושלא ("and that ... not")."""

    for run in GRAMMATICAL_CHAIN_RUNS:
        for body in ("לא חסר סעיף 2", "זולת חסר סעיף 2"):
            assert _issues(pinned.REVIEW_SOURCE, run + joint + body) == _issues(
                pinned.REVIEW_SOURCE, run + body
            ), (run, joint, body)
    assert pinned._accepted("a", "ומ" + joint + "לא חסר סעיף 2", pinned.REVIEW_SOURCE)


@pytest.mark.parametrize("joint", JOINTS)
def test_refusal_readers_restart_after_runs_grammar_cannot_stack(joint):
    """An ambiguous run has two readings of its dash: refusal reads both.

    Its joined spelling still reads (דמ־לבד keeps מלבד, "except"; וו־אילו
    keeps וואילו, "and whereas"), and a standalone negator or adversative
    after the dash is visible too.
    """

    for run in AMBIGUOUS_RUNS:
        for body in ("לא חסר סעיף 2", "אין זה נכון שחסר סעיף 2"):
            text = hebrew.fold_marks(run + joint + body)
            start = text.index(body)
            for pattern in (hebrew._REASON_NEGATION, hebrew._NEGATIONS):
                assert any(m.start() == start for m in pattern.finditer(text)), (
                    run,
                    body,
                )
            assert not pinned._accepted("a", run + joint + body, pinned.REVIEW_SOURCE)
        assert not pinned._accepted(
            "a", f"חסר סעיף 2. {run}{joint}אולם הוא זמין.", pinned.REVIEW_SOURCE
        ), run
        source = f"(א) מס על {run}{joint}לא מחושב בהתאם לסעיף 2."
        assert not pinned._accepted("a", "חסר סעיף 2", source), run
        if run.endswith("מ"):
            source = f"(א) מס על {run}{joint}לבד חישוב בהתאם לסעיף 2."
            assert not pinned._accepted("a", "חסר סעיף 2", source), run
    assert not pinned._accepted(
        "a", f"חסר סעיף 2. וו{joint}אילו הוא זמין.", pinned.REVIEW_SOURCE
    )


def test_prefixed_negators_spelled_in_prefix_letters_end_at_the_dash():
    """ובל ("and not"), שבל and דבל ("that not") keep their own prefix.

    No reader crosses the punctuation after these word runs, but a match that
    ends at the dash is no crossing: glued, dashed and spaced spellings agree.
    """

    runs = {
        prefix + form
        for family in NEGATOR_FAMILIES
        for prefix in family.prefixes
        for form in family.forms
        if prefix
        and set(prefix + form) <= set(HEBREW_PREFIX_LETTERS)
        and len(prefix + form) <= 3
    }
    assert runs == {"ובל", "שבל", "דבל"}
    assert not runs & PREFIX_CHAIN_RUNS
    for run, joint in itertools.product(sorted(runs), ("\u05be", "-", " - ", " ")):
        for text in (run + joint + "נכון שחסר סעיף 2", run + joint + "מחושב"):
            folded = hebrew.fold_marks(text)
            for pattern in (hebrew._REASON_NEGATION, hebrew._NEGATIONS):
                match = pattern.search(folded)
                assert match is not None and match.span() == (0, 3), (run, joint)
        assert not pinned._accepted(
            "a", run + joint + "נכון שחסר סעיף 2", pinned.REVIEW_SOURCE
        )


@pytest.mark.parametrize("joint", JOINTS)
def test_no_reader_crosses_the_dash_after_a_word_run(joint):
    """של־לא is "of", a dash and לא: never ש with ללא ("that without")."""

    for run in WORD_RUNS:
        reason = run + joint + "לא חסר סעיף 2"
        assert not pinned._accepted("a", reason, pinned.REVIEW_SOURCE), (run, joint)
        text = hebrew.fold_marks(reason)
        start = text.index("לא חסר")
        for match in hebrew._SIGNAL.finditer(text):
            assert match.start() >= start, (run, joint, match.group())
