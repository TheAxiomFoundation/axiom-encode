"""Visible prefix joints are grammatical punctuation in every Hebrew reader."""

import itertools
import json
import re
import sys
import tracemalloc
import unicodedata
from functools import lru_cache
from pathlib import Path

import pytest

from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness.hebrew_negators import NEGATOR_FAMILIES
from axiom_encode.harness.hebrew_prefixes import (
    PREFIX_CORPUS_ABBREVIATIONS,
    PREFIX_JOINT_CHARS,
    PREFIX_WHOLE_WORDS,
    STATUTORY_PREFIX_WHOLE_WORDS,
    compact_prefix_joints,
    joins_at_dash,
    prefix_chain,
    prefix_word,
)
from tests import test_source_completeness_hebrew as pinned

JOINTS = tuple(PREFIX_JOINT_CHARS)
SPACED_JOINTS = tuple(
    " " * left + dash + " " * right
    for dash, left, right in itertools.product(JOINTS, range(3), range(3))
)
PREFIX_LETTERS = "והבכלמשד"
PREFIXES = tuple(
    "".join(chain)
    for size in range(4)
    for chain in itertools.product(PREFIX_LETTERS, repeat=size)
)
# Every run of prefix letters a dash leaves as punctuation (round 17): the
# explicit words and every run grammar cannot stack as proclitics.
WORD_RUNS = tuple(prefix for prefix in PREFIXES if prefix and not joins_at_dash(prefix))


def _joint_chain(prefix: str, joint: str) -> str:
    return joint.join(prefix) + joint if prefix else ""


def _issues(source: str, reason: str) -> tuple[str, ...]:
    return pinned._analyze(pinned._deferral_content("a", reason), source)


@lru_cache(maxsize=1024)
def _glued_issues(source: str, reason: str) -> tuple[str, ...]:
    # Every joint compares against these identical glued controls. Reuse their
    # pure diagnostic tuples; each distinct joined input still runs the full
    # analyzer. The bounded cache avoids repeating YAML analysis 252 times.
    return _issues(source, reason)


_SOURCE_LOCATOR = re.compile(
    r"Source locator at characters (?P<start>\d+):(?P<end>\d+): `(?P<excerpt>.*?)`\."
)


def _assert_source_issue_parity(source, joined, split, reason, expected):
    """Compare exact diagnostics after validating original-source provenance.

    Formula diagnostics report original spans and whitespace-collapsed source
    previews. Validate each preview against its original slice, remove only the
    inserted joint from that raw slice, map its locator back, and compare all
    remaining diagnostic characters. The display's whitespace policy is the
    existing formatter's; source offsets still refer to the untouched text.
    """

    def locators(issue, original, insertion=None):
        def mapped(match):
            start, end = int(match["start"]), int(match["end"])
            excerpt = match["excerpt"]
            raw = original[start:end]
            assert " ".join(raw.split()) == excerpt, (original, match.group())
            if insertion is None:
                return match.group()
            width = len(joined) - len(source)
            if start <= insertion < end:
                offset = insertion - start
                inserted = raw[offset : offset + width]
                assert inserted.strip() in JOINTS
                raw = raw[:offset] + raw[offset + width :]
            start -= width if start > insertion else 0
            end -= width if end > insertion else 0
            assert source[start:end] == raw
            excerpt = " ".join(raw.split())
            return f"Source locator at characters {start}:{end}: `{excerpt}`."

        return _SOURCE_LOCATOR.sub(mapped, issue)

    validated = tuple(locators(issue, source) for issue in expected)
    mapped = tuple(locators(issue, joined, split) for issue in _issues(joined, reason))
    assert mapped == validated, joined


def test_prefix_utility_enumerates_every_chain_and_visible_joint():
    """All 585 literal chains accept every Unicode dash without deletion."""

    pattern = re.compile(prefix_chain(PREFIX_LETTERS, 3) + "סעיף")
    for prefix, joint in itertools.product(PREFIXES, JOINTS):
        text = _joint_chain(prefix, joint) + "סעיף"
        assert pattern.fullmatch(text), text
        assert hebrew.fold_marks(text) == text
        assert re.fullmatch(prefix_word(prefix) + "סעיף", text), text


@pytest.mark.parametrize("joint", JOINTS)
def test_leading_joint_is_punctuation_and_never_hides_a_standalone_negator(joint):
    for text in (joint + "לא חסר סעיף 2", "מידע" + joint + "לא חסר סעיף 2"):
        assert hebrew._REASON_NEGATION.search(text), text
        assert _issues(pinned.REVIEW_SOURCE, text)


@pytest.mark.parametrize("joint", JOINTS)
def test_negator_joint_family_preserves_glued_readings_and_lexical_exclusions(joint):
    """Only grammatical prefixes split; lexical compounds retain their roots.

    In particular, לולא is a conditional word, not the prohibited prefix לו
    attached to לא. Its own allowed conjunction/relative prefixes are tested.
    Real complete prefix-only words have punctuation controls separately.
    """

    cases = {
        (prefix, form)
        for family in NEGATOR_FAMILIES
        for prefix in family.prefixes
        for form in family.forms
        if prefix not in PREFIX_WHOLE_WORDS
    }
    # These deliberately invalid prepositional/article splits retain their
    # lexical exclusions: מלא "full", כלא "prison", הלא "surely", כבל "cable",
    # מאין "whence". They must never restart at the standalone suffix.
    cases.update((("מ", "לא"), ("כ", "לא"), ("ה", "לא"), ("כ", "בל"), ("מ", "אין")))
    for prefix, form in sorted(cases):
        glued = prefix + form + " חסר סעיף 2"
        expected = bool(hebrew._REASON_NEGATION.search(glued))
        assert bool(hebrew._NEGATIONS.search(glued)) == expected, glued
        text = _joint_chain(prefix, joint) + form + " חסר סעיף 2"
        assert bool(hebrew._REASON_NEGATION.search(text)) == expected, (glued, text)
        assert bool(hebrew._NEGATIONS.search(text)) == expected, (glued, text)


def test_signal_prefixes_preserve_classification_and_article_noun_exclusions():
    forms = (
        "חסר סעיף 2",
        "חסרה הוראה",
        "חסרים פרטים",
        "חסרות הוראות",
        "טרם קודד",
        "עדיין לא קודד",
        "בלתי זמין",
        "דורש סעיף 2",
        "תלוי בסעיף 2",
        "עד שיקודד",
        "עד אשר יקודד",
        "בלי סעיף 2",
        "ללא סעיף 2",
        "מבלי סעיף 2",
        "בלא סעיף 2",
        "לא ניתן לחשב",
        "אי אפשר לקודד",
    )
    for prefix, form in itertools.product(("ו", "ש", "ה", "וש", "שה", "וה"), forms):
        glued = prefix + form
        baseline = hebrew._SIGNAL.search(glued)
        expected = baseline.lastgroup if baseline else None
        for joint in JOINTS:
            joined = _joint_chain(prefix, joint) + form
            actual = hebrew._SIGNAL.search(joined)
            assert (actual.lastgroup if actual else None) == expected, (glued, joined)


@pytest.mark.parametrize("joint", JOINTS)
@pytest.mark.parametrize(
    "text,boundary",
    (("תלוי בסעיף 2", 6), ("עד שיקודד", 4), ("לא ניתן לחשב", 9), ("אי אפשר לקודד", 9)),
)
def test_signal_internal_prefixes_keep_their_classification(joint, text, boundary):
    baseline = hebrew._SIGNAL.search(text)
    assert baseline is not None
    joined = text[:boundary] + joint + text[boundary:]
    actual = hebrew._SIGNAL.search(joined)
    assert actual is not None, joined
    assert actual.lastgroup == baseline.lastgroup


@pytest.mark.parametrize("joint", JOINTS)
@pytest.mark.parametrize("exclusion,prefix", (("חוץ ", "מ"), ("פרט ", "ל")))
@pytest.mark.parametrize("object_word", ("תשלום", "חישוב", "calculation", "Zahlung"))
def test_exclusion_prefix_joint_preserves_full_pipeline_issues(
    joint, exclusion, prefix, object_word
):
    glued = f"(א) כל התשלומים, {exclusion}{prefix}{object_word} בהתאם לסעיף 2."
    joined = f"(א) כל התשלומים, {exclusion}{prefix}{joint}{object_word} בהתאם לסעיף 2."
    expected = _issues(glued, "חסר סעיף 2")
    assert expected, glued
    assert _issues(joined, "חסר סעיף 2") == expected


@pytest.mark.parametrize("joint", ("", *JOINTS))
@pytest.mark.parametrize("exclusion,prefix", (("חוץ ", "מ"), ("פרט ", "ל")))
def test_extra_german_computation_object_is_always_refused(joint, exclusion, prefix):
    """The generic German computation diagnostic can additionally read the joint.

    The pinned cross-language table separates ``Berechnung`` from its negator.
    This extra compound glues it to the exclusion's preposition.  Its deferral
    always fails; with visible punctuation the independent German formula
    reader can also recognize ``Berechnung`` and emit a formula-output issue.
    """

    source = f"(א) כל התשלומים, {exclusion}{prefix}{joint}Berechnung בהתאם לסעיף 2."
    assert not pinned._accepted("a", "חסר סעיף 2", source)


# These are prefix boundaries within the lexical spellings of exclusions,
# linkers, signals and computations, rather than before their whole word.
LEXICAL_PREFIX_CONSTRUCTIONS = (
    ("reason", "ו", "חסר סעיף 2"),
    ("reason", "ש", "חסר סעיף 2"),
    ("reason", "שה", "חסר סעיף 2"),
    ("reason", "ו", "ללא סעיף 2"),
    ("reason", "מ", "בלי סעיף 2"),
    ("reason", "ל", "לא סעיף 2"),
    ("source", "ל", "מעט חישוב בהתאם לסעיף 2"),
    ("source", "ל", "הוציא חישוב בהתאם לסעיף 2"),
    ("source", "מ", "לבד חישוב בהתאם לסעיף 2"),
    ("source", "מ", "בלי חישוב בהתאם לסעיף 2"),
    ("source", "ל", "לא חישוב בהתאם לסעיף 2"),
    ("source", "ב", "לא חישוב בהתאם לסעיף 2"),
    ("source", "ב", "התאם לסעיף 2"),
    ("source", "ל", "פי סעיף 2"),
)


@pytest.mark.parametrize("joint", JOINTS)
@pytest.mark.parametrize("position,prefix,body", LEXICAL_PREFIX_CONSTRUCTIONS)
def test_lexical_prefix_constructions_preserve_full_pipeline_issues(
    joint, position, prefix, body
):
    glued = prefix + body
    joined = _joint_chain(prefix, joint) + body
    if position == "source":
        assert _issues(f"(א) {joined}.", "חסר סעיף 2") == _issues(
            f"(א) {glued}.", "חסר סעיף 2"
        )
    else:
        assert _issues(pinned.REVIEW_SOURCE, joined) == _issues(
            pinned.REVIEW_SOURCE, glued
        )


@pytest.mark.parametrize("joint", JOINTS)
@pytest.mark.parametrize("prefix", ("ל", "ב", "שב", "ושל"))
def test_citation_readers_preserve_paths_and_original_offsets(joint, prefix):
    glued = prefix + "סעיף 14(א)"
    joined = _joint_chain(prefix, joint) + "סעיף 14(א)"
    assert hebrew.cites_current_branch(joined, pinned.CITATION, ("a",))
    assert hebrew.cites_current_section(joined, pinned.CITATION)
    assert hebrew.native_source_reference_paths(joined, pinned.CITATION) == {("a",)}
    for text in (glued, joined):
        dependency = hebrew.SECTION_DEPENDENCY.fullmatch(text)
        assert dependency is not None, text
        assert text[dependency.end("proclitics") :] == "סעיף 14(א)"


@pytest.mark.parametrize("joint", JOINTS)
@pytest.mark.parametrize("base", hebrew._DEFINITE_INSTRUMENT_BASES)
def test_known_instrument_articles_preserve_parsed_identity_and_source_text(
    joint, base
):
    glued = "ה" + base
    joined = "ה" + joint + base
    assert hebrew.fold_marks(joined) == joined
    assert hebrew._instrument_words(joined) == hebrew._instrument_words(glued)
    source = "חוק הביטוח הלאומי; סעיף 2 לחוק "
    expected = hebrew._section_citations(source + glued)
    actual = hebrew._section_citations(source + joined)
    assert len(actual) == len(expected) == 1
    assert actual[0][1] == expected[0][1]
    assert actual[0][0].group("instrument_name").strip() == joined


@pytest.mark.parametrize("joint", JOINTS)
@pytest.mark.parametrize("base", ("זה", "זאת", "אלה"))
def test_current_instrument_article_joints_keep_current_branch_ownership(joint, base):
    text = f"סעיף 14(א) לחוק ה{joint}{base}"
    assert hebrew.cites_current_branch(text, pinned.CITATION, ("a",))
    assert hebrew.cites_current_section(text, pinned.CITATION)
    assert hebrew.native_source_reference_paths(text, pinned.CITATION) == {("a",)}


def test_all_adversative_prefixes_and_glued_endings_preserve_issue_strings():
    """Every word, phrase, conjunction and final glued prefix is enumerated."""

    constructions = [
        *hebrew._ADVERSATIVE_WORDS.split(),
        *hebrew._ADVERSATIVE_PHRASES,
        *(phrase + " " + glue + "מידע" for phrase, glue in hebrew._ADVERSATIVE_GLUED),
        "להוציא מידע",
        "בעוד מידע",
        "אך מידע",
    ]
    checked = 0
    for construction in constructions:
        for conjunction in ("", "ו"):
            term = conjunction + construction
            boundaries = {1} if conjunction else set()
            for match in re.finditer(r"[א-ת]+", construction):
                if match.group() in hebrew._ADVERSATIVE_PREFIX_WORDS:
                    boundaries.add(len(conjunction) + match.start() + 1)
            for phrase, glue in hebrew._ADVERSATIVE_GLUED:
                if construction == phrase + " " + glue + "מידע":
                    boundaries.add(len(conjunction) + len(phrase) + 1 + len(glue))
            if construction.startswith(("להוציא", "בעוד")):
                boundaries.add(len(conjunction) + 1)
            expected = _issues(pinned.REVIEW_SOURCE, "חסר סעיף 2. " + term)
            for boundary, joint in itertools.product(boundaries, JOINTS):
                joined = term[:boundary] + joint + term[boundary:]
                assert (
                    _issues(pinned.REVIEW_SOURCE, "חסר סעיף 2. " + joined) == expected
                ), joined
                checked += 1
    assert checked >= 3


def _pinned_rows():
    for round_number in (12, 13, 14, 15, 16):
        for index, row in enumerate(
            getattr(pinned, f"ROUND_{round_number}_REPRODUCERS", ())
        ):
            if len(row) == 5:
                source, _english_source, reason, _english_reason, _verdict = row
            else:
                _route, source, _english_source, reason, _english_reason, _verdict = row
            yield round_number, index, source, reason
    for index, row in enumerate(pinned.ROUND_15_NEGATOR_TRANSLATIONS):
        _form, reason, _english_reason, source, _english_source, *_verdicts = row
        yield 15, ("negator", index), source, reason


# A prefix is split only before a base the harness actually reads.  Splitting
# arbitrary Hebrew words would invent new words (``מלא`` is the noun "full").
_CLASS_BASES = "|".join(
    re.escape(form)
    for form in sorted(
        {form for family in NEGATOR_FAMILIES for form in family.forms},
        key=lambda value: (-len(value), value),
    )
)
_PINNED_PREFIX = re.compile(
    r"(?<![א-ת])(?P<prefix>[והבכלמשד]{1,3})"
    rf"(?P<base>{_CLASS_BASES}|חסר(?:ה|ים|ות)?|סעיף"
    r"|not|nicht|however|prohibited|בהתאם|לפי|אולם|אבל|בלי|מעט|לבד|הוציא"
    r"|התאם|פי)(?![א-תA-Za-z])"
)


@pytest.mark.parametrize("joint", SPACED_JOINTS)
def test_every_pinned_prefixed_construction_preserves_verdict_and_issue_strings(joint):
    """Replay rounds 12–16 under every dash and zero to two spaces per side."""

    checked = set()
    for round_number, index, source, reason in _pinned_rows():
        for position, text in (("source", source), ("reason", reason)):
            for match in _PINNED_PREFIX.finditer(hebrew.fold_marks(text)):
                if not joins_at_dash(match["prefix"]):
                    continue
                # The tables also contain invisible characters.  Their fold
                # has its own exhaustive property; use this folded spelling
                # as the glued control for this visible-punctuation property.
                folded = hebrew.fold_marks(text)
                expected = (
                    _glued_issues(folded, reason)
                    if position == "source"
                    else _glued_issues(source, folded)
                )
                split = match.end("prefix")
                variant = folded[:split] + joint + folded[split:]
                if position == "source":
                    _assert_source_issue_parity(
                        folded, variant, split, reason, expected
                    )
                else:
                    assert _issues(source, variant) == expected, (
                        round_number,
                        index,
                        position,
                        joint,
                        variant,
                    )
                checked.add((round_number, index, position, split, joint))
    assert len(checked) >= 20


@pytest.mark.parametrize("joint", SPACED_JOINTS)
def test_pinned_cross_language_tables_preserve_every_prefix_joint(joint):
    """Every pinned entry meets the joints, with original locators verified.

    The existing translation property tests the complete Cartesian matrix.
    Here a covering set visits every negator, object and link while avoiding
    thousands of duplicate integration checks for the same prefix grammar.
    """

    negator = next(word for word in pinned._CROSS_NEGATORS if word.text == "מבלי")
    obj = pinned._CROSS_OBJECTS[0]
    link = pinned._CROSS_LINKS[0]
    frames = {
        *((word, obj, link) for word in pinned._CROSS_NEGATORS),
        *((negator, word, link) for word in pinned._CROSS_OBJECTS),
        *((negator, obj, word) for word in pinned._CROSS_LINKS),
    }
    assert {frame[0] for frame in frames} == set(pinned._CROSS_NEGATORS)
    assert {frame[1] for frame in frames} == set(pinned._CROSS_OBJECTS)
    assert {frame[2] for frame in frames} == set(pinned._CROSS_LINKS)
    checked = 0
    for negator, obj, link in sorted(frames):
        text = f"(א) משולם {negator.text} {obj.text} {link.text}."
        matches = tuple(
            match
            for match in _PINNED_PREFIX.finditer(text)
            if joins_at_dash(match["prefix"])
        )
        if not matches:
            continue
        expected = _glued_issues(text, "חסר סעיף 2")
        for match in matches:
            split = match.end("prefix")
            joined = text[:split] + joint + text[split:]
            _assert_source_issue_parity(text, joined, split, "חסר סעיף 2", expected)
            checked += 1
    assert checked >= 3


# Python exposes Pd directly. The additional mathematical minus belongs to
# the user-specified closed joint class; maqaf/FE63/FF0D are already Pd.
def test_joint_class_is_every_unicode_dash_and_the_mathematical_minus():
    expected = {
        chr(code)
        for code in range(sys.maxunicode + 1)
        if unicodedata.category(chr(code)) == "Pd"
    } | {"\u05be", "\u2212", "\ufe63", "\uff0d"}
    assert set(PREFIX_JOINT_CHARS) == expected


@pytest.mark.parametrize("dash", JOINTS)
@pytest.mark.parametrize("left,right", itertools.product(range(3), repeat=2))
def test_round16_exclusion_and_citation_joints_match_glued_pipeline(dash, left, right):
    joint = " " * left + dash + " " * right
    for prefix, body in (("מ", "תשלום"), ("מ", "חישוב")):
        glued = f"(א) חוץ {prefix}{body} בהתאם לסעיף 2."
        joined = f"(א) חוץ {prefix}{joint}{body} בהתאם לסעיף 2."
        assert _issues(joined, "חסר סעיף 2") == _issues(glued, "חסר סעיף 2")
    glued = "(א) ישולם בהתאם לסעיף 2."
    joined = f"(א) ישולם בהתאם ל{joint}סעיף 2."
    assert _issues(joined, "חסר סעיף 2") == _issues(glued, "חסר סעיף 2")


@pytest.mark.parametrize("dash", JOINTS)
@pytest.mark.parametrize("left,right", itertools.product(range(3), repeat=2))
def test_real_dash_punctuation_keeps_its_source_reading(dash, left, right):
    joint = " " * left + dash + " " * right
    for before, after in (
        ("2", "4"),
        ("סעיף 2", "חסר"),
        ("תשלום", "חסר סעיף 2"),
        *((word, "תשלום") for word in WORD_RUNS),
    ):
        text = before + joint + after
        assert compact_prefix_joints(text) == text
        assert hebrew.fold_marks(text) == text


def test_joint_gaps_are_horizontal_and_preserve_all_original_offsets():
    horizontal = "\t \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
    for gap in horizontal:
        source = "(א) מס ל" + gap * 4 + "־" + gap * 5 + "סעיף 2."
        folded = hebrew.FoldedText(source)
        assert folded.text == "(א) מס ל־סעיף 2."
        assert folded.text == hebrew.fold_marks(source)
        for offset, char in enumerate(folded.text):
            assert source[folded.original_start(offset)] == char
            assert source[folded.original_end(offset + 1) - 1] == char
    for gap in ("\n", "\r", "\v", "\f", "\x85", "\u2028", "\u2029"):
        text = "ל" + gap + "־" + gap + "סעיף 2"
        assert compact_prefix_joints(text) == text


def test_default_prefix_chain_is_bounded():
    pattern = re.compile(prefix_chain(PREFIX_LETTERS) + "בלתי")
    assert pattern.fullmatch("ושדבלתי")
    assert pattern.fullmatch("ושדמבלתי") is None


def test_one_million_prefix_letters_has_bounded_memory():
    """Run this stress property under run_capped.py (2 GiB / 600 seconds)."""

    # The Unicode property table is a one-time shared cache, not per-input
    # regex state. Warm it before measuring growth on the million-letter run.
    hebrew.fold_marks("ל־סעיף 2")
    text = (PREFIX_LETTERS * 125_000) + "בלתי חסר סעיף 2"
    tracemalloc.start()
    try:
        folded = hebrew.fold_marks(text)
        assert folded is text
        assert hebrew._REASON_NEGATION.search(folded) is None
        assert hebrew._NEGATIONS.search(folded) is None
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024, peak


def test_prefix_only_corpus_tokens_are_words_or_explicit_abbreviations():
    corpus = json.loads(
        (Path(__file__).parent / "fixtures" / "hebrew_negator_corpus.json").read_text()
    )
    tokens = {
        word
        for word in corpus["vocabulary"]
        if 2 <= len(word) <= 3 and set(word) <= set(PREFIX_LETTERS)
    }
    assert (
        tokens == (STATUTORY_PREFIX_WHOLE_WORDS - {"משה"}) | PREFIX_CORPUS_ABBREVIATIONS
    )
    assert not STATUTORY_PREFIX_WHOLE_WORDS & PREFIX_CORPUS_ABBREVIATIONS
    assert STATUTORY_PREFIX_WHOLE_WORDS <= PREFIX_WHOLE_WORDS


@pytest.mark.parametrize("joint", SPACED_JOINTS)
def test_dash_after_a_full_word_keeps_negators_visible_in_both_guards(joint):
    for word in ("כל", "טל", "חשמל", "תשלום"):
        reason = word + joint + "לא חסר סעיף 2"
        source = "(א) " + word + joint + "לא מחושב בהתאם לסעיף 2."
        assert _issues(pinned.REVIEW_SOURCE, reason) == _issues(
            pinned.REVIEW_SOURCE, word + " לא חסר סעיף 2"
        )
        assert _issues(source, "חסר סעיף 2") == _issues(
            "(א) " + word + " לא מחושב בהתאם לסעיף 2.", "חסר סעיף 2"
        )
        assert hebrew._REASON_NEGATION.search(hebrew.fold_marks(reason))
        assert hebrew._NEGATIONS.search(hebrew.fold_marks(source))


@pytest.mark.parametrize("joint", SPACED_JOINTS)
def test_every_whole_word_keeps_negation_article_and_relative_readings(joint):
    """All 421 word runs meet every dash in every Hebrew reader.

    A positive preceding word never becomes a relative prefix, and its final
    ה never becomes the article of the following feminine signal/exclusion.
    The non-prefix-alphabet controls cover the same punctuation rule.
    """

    words = set(WORD_RUNS) | {"חיה", "תקופה", "טל", "חשמל", "תשלום"}
    for word in sorted(words):
        for body in (
            "לא מחושב לפי סעיף 2",
            "בלתי מחושב לפי סעיף 2",
            "זולת תשלום לפי סעיף 2",
        ):
            text = hebrew.fold_marks(word + joint + body)
            start = text.index(body)
            for pattern in (hebrew._NEGATIONS, hebrew._REASON_NEGATION):
                roots = [
                    match for match in pattern.finditer(text) if match.start() == start
                ]
                assert roots, (word, joint, body, text)
                assert all(not hebrew._is_relative(match) for match in roots), text
        for body in ("חסרה הוראה", "חסרות הוראות"):
            control = hebrew._SIGNAL.search(word + " " + body)
            actual = hebrew._SIGNAL.search(hebrew.fold_marks(word + joint + body))
            assert control is not None
            assert actual is not None, (word, joint, body)
            assert actual.lastgroup == control.lastgroup, (word, joint, body)


@pytest.mark.parametrize("joint", SPACED_JOINTS)
def test_full_word_punctuation_preserves_pipeline_relative_and_article_scope(joint):
    for word in ("דוד", "בד", "כלב", "מה", "שמה", "חיה", "תקופה"):
        for body in ("בלתי מחושב לפי סעיף 2", "זולת תשלום לפי סעיף 2"):
            assert _issues(f"(א) מס על {word}{joint}{body}.", "חסר סעיף 2") == _issues(
                f"(א) מס על {word} {body}.", "חסר סעיף 2"
            ), (word, joint, body)
        for body in ("בלתי נכון שחסר סעיף 2", "חסרה הוראה בסעיף 2"):
            assert _issues(pinned.REVIEW_SOURCE, word + joint + body) == _issues(
                pinned.REVIEW_SOURCE, word + " " + body
            ), (word, joint, body)


@pytest.mark.parametrize("joint", SPACED_JOINTS)
def test_forward_prefix_guard_preserves_negative_core_words_before_punctuation(joint):
    for core in ("בל", "לבל", "כשל"):
        text = hebrew.fold_marks(core + joint + "מחושב לפי סעיף 2")
        for pattern in (hebrew._NEGATIONS, hebrew._REASON_NEGATION):
            match = pattern.search(text)
            assert match is not None, (core, joint)
            assert match.start() == 0
            assert not hebrew._is_relative(match)
