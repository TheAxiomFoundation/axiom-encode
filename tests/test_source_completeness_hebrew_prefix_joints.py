"""Visible prefix joints are grammatical punctuation in every Hebrew reader."""

import itertools
import re

import pytest

from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness.hebrew_prefixes import (
    PREFIX_JOINT_CHARS,
    prefix_chain,
    prefix_word,
)
from tests import test_source_completeness_hebrew as pinned

JOINTS = tuple(PREFIX_JOINT_CHARS)
PREFIX_LETTERS = "והבכלמשד"
PREFIXES = tuple(
    "".join(chain)
    for size in range(4)
    for chain in itertools.product(PREFIX_LETTERS, repeat=size)
)


def _joint_chain(prefix: str, joint: str) -> str:
    return joint.join(prefix) + joint if prefix else ""


def _issues(source: str, reason: str) -> tuple[str, ...]:
    return pinned._analyze(pinned._deferral_content("a", reason), source)


_SOURCE_LOCATOR = re.compile(
    r"Source locator at characters (?P<start>\d+):(?P<end>\d+): `(?P<excerpt>.*?)`\."
)


def _assert_source_issue_parity(source, joined, split, reason, expected):
    """Compare exact diagnostics after validating original-source provenance.

    Formula diagnostics deliberately quote the verbatim source and its character
    span. Inserting visible punctuation must change those fields. Validate each
    original slice, map only that locator back across the one inserted joint,
    and compare every remaining character of every issue without normalization.
    """

    def locators(issue, original, insertion=None):
        def mapped(match):
            start, end = int(match["start"]), int(match["end"])
            excerpt = match["excerpt"]
            assert original[start:end] == excerpt, (original, match.group())
            if insertion is None:
                return match.group()
            if start <= insertion < end:
                offset = insertion - start
                assert excerpt[offset] in JOINTS
                excerpt = excerpt[:offset] + excerpt[offset + 1 :]
            start -= start > insertion
            end -= end > insertion
            assert source[start:end] == excerpt
            return f"Source locator at characters {start}:{end}: `{excerpt}`."

        return _SOURCE_LOCATOR.sub(mapped, issue)

    validated = tuple(locators(issue, source) for issue in expected)
    mapped = tuple(locators(issue, joined, split) for issue in _issues(joined, reason))
    assert mapped == validated, joined


def test_prefix_utility_enumerates_every_chain_and_visible_joint():
    """All 585 prefix chains accept all four joints without deleting them."""

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


def test_negator_joint_family_preserves_glued_readings_and_lexical_exclusions():
    """Both readers retain the shared lexicon under every prefix and joint."""

    forms = (
        "לא",
        "אין",
        "אינו",
        "אינה",
        "אינם",
        "אינן",
        "איני",
        "אינך",
        "אינכם",
        "אינכן",
        "איננו",
        "איננה",
        "אינני",
        "בלתי",
        "טרם",
        "מעולם",
        "לאו",
        "אל תטען",
        "בלי",
        "מבלי",
        "ללא",
        "בלא",
        "אף",
        "אסור",
        "נאסר",
        "מסרב",
        "למעט",
        "מלבד",
        "להוציא",
        "זולת",
    )
    for prefix, form in itertools.product(PREFIXES, forms):
        glued = prefix + form + " חסר סעיף 2"
        expected = bool(hebrew._REASON_NEGATION.search(glued))
        assert bool(hebrew._NEGATIONS.search(glued)) == expected, glued
        for joint in JOINTS:
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
    assert checked >= 300


def _pinned_rows():
    for round_number in (12, 13, 14, 15):
        for index, row in enumerate(
            getattr(pinned, f"ROUND_{round_number}_REPRODUCERS", ())
        ):
            if len(row) == 5:
                source, _english_source, reason, _english_reason, _verdict = row
            else:
                _route, source, _english_source, reason, _english_reason, _verdict = row
            yield round_number, index, source, reason


# A prefix is split only before a base the harness actually reads.  Splitting
# arbitrary Hebrew words would invent new words (``מלא`` is the noun "full").
_PINNED_PREFIX = re.compile(
    r"(?<![א-ת])(?P<prefix>[והבכלמשד]{1,3})"
    r"(?P<base>בלתי|לא|אין|אינו|אינה|אינם|אינן|טרם|חסר(?:ה|ים|ות)?|סעיף"
    r"|not|nicht|however|prohibited|בהתאם|לפי|אולם|אבל|בלי|מעט|לבד|הוציא"
    r"|התאם|פי)(?![א-תA-Za-z])"
)


def test_every_pinned_prefixed_construction_preserves_verdict_and_issue_strings():
    """Replay rounds 12–15, retaining correct original-source locator fields."""

    checked = set()
    for round_number, index, source, reason in _pinned_rows():
        for position, text in (("source", source), ("reason", reason)):
            for match in _PINNED_PREFIX.finditer(hebrew.fold_marks(text)):
                # The tables also contain invisible characters.  Their fold
                # has its own exhaustive property; use this folded spelling
                # as the glued control for this visible-punctuation property.
                folded = hebrew.fold_marks(text)
                expected = (
                    _issues(folded, reason)
                    if position == "source"
                    else _issues(source, folded)
                )
                split = match.end("prefix")
                for joint in JOINTS:
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
    assert len(checked) >= 80


def test_pinned_cross_language_tables_preserve_every_prefix_joint():
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
        matches = tuple(_PINNED_PREFIX.finditer(text))
        if not matches:
            continue
        expected = _issues(text, "חסר סעיף 2")
        for match, joint in itertools.product(matches, JOINTS):
            split = match.end("prefix")
            joined = text[:split] + joint + text[split:]
            _assert_source_issue_parity(text, joined, split, "חסר סעיף 2", expected)
            checked += 1
    assert checked >= 300
