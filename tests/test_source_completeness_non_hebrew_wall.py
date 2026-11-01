"""The non-Hebrew wall (#1882, round 21).

A deferral whose reason and source hold no Hebrew-block code point, raw or
folded, takes main's verdict and main's issue strings, byte for byte. German
text counts as non-Hebrew. The Hebrew additions reach such text through
shared English patterns, so these tests hold the wall at each of them:

- the English link guard reads a raw scope with main's own pattern, pinned
  here by digest, and the German words of ``_SOURCE_LINK_NEGATION_LANGUAGE``
  only in the cross-language copy of a scope that holds Hebrew;
- each shared English pattern that gained Hebrew alternatives is main's
  pattern, pinned by digest, plus a suffix that matches nothing without a
  Hebrew letter;
- every literal alternative this change added to an English-reachable
  pattern, read from the compiled patterns themselves, is injected as a name
  into English and German frames, where it must change nothing that the
  neutral name ``Lee`` does not. No main checkout exists in CI, so the neutral
  frame stands in for main; the round-21 replay ties that frame to main
  (``.review-scratch/round21/REPORT.md``);
- main's exact issue arrays pin the review's finding 5 and 6 inputs.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import re
import unicodedata
from pathlib import Path
from re import _constants as sre_constants
from re import _parser as sre_parser

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import hebrew_source_text
from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

MAIN_ISSUES = Path(__file__).parent / "fixtures/non_hebrew_wall_main_issues.json"
MAIN = "5d40209bc0e1fea5f740581ad5bb1b114ceeff07"
DEFERRAL = "[complete-source-unit:deferral]"
STRUCTURE = "[complete-source-unit:structure]"

# Main's English link negations, the pattern literal of
# ``_source_link_scope_has_unreset_negation`` at MAIN.
MAIN_ENGLISH_LINK_NEGATION_SHA256 = (
    "f58170d9e645a1a5d032fee6bf167494961294dff04df5b187c5c24fcca93772"
)
# Main's pattern strings of the shared English patterns that gained Hebrew
# alternatives, with their digests at MAIN.
MAIN_SHARED_PATTERNS = {
    "_MISSING_DEPENDENCY_LANGUAGE": (
        r"\b(?:requires?|depends?\s+on|missing|not\s+yet\s+encoded|unavailable|"
        r"cannot\s+be\s+(?:computed|encoded|resolved)|until|without|"
        r"benötigt|abhängig|fehlt|nicht\s+codiert)\b",
        "2bf4250741ba58404a7909f1d686b02461c55fb341b9928b912e63c7f2a9e831",
        hebrew_source_text.MISSING_DEPENDENCY_LANGUAGE,
    ),
    "_ADVERSATIVE_LANGUAGE": (
        r"\b(?:although|but|despite|even\s+though|except|however|nevertheless|"
        r"nonetheless|notwithstanding|though|unless|whereas|while|yet|aber|"
        r"jedoch|obwohl)\b",
        "9efadff1b00509fc5b44ebefc2cb1f962cb60c6c45a6a72e76474bf63e335996",
        hebrew_source_text.ADVERSATIVE_LANGUAGE,
    ),
    "_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE": (
        r"\b(?:administrative|calendar|capabilit(?:y|ies)|classification|"
        r"complaint|content|data|date\s+arithmetic|designation|determination|"
        r"document|event|fact|filing|form|hearing|information|input|"
        r"investigation|membership|notice|process|procedure|record|relation|"
        r"representation|status|submission|timing|workflow)[a-z-]*\b",
        "cd6ada7337ae714361893fa7c85a8176f046aeb4dcbbdda051ac2eebc0210064",
        hebrew_source_text.RUNTIME_GAP_LANGUAGE,
    ),
}
MAIN_EDITORIAL_OMISSION_ONLY = (
    r"^\s*(?:(?:\((?:\d+[a-z]?|[a-z])\)|\d+[a-z]?\.|[a-z]\))\s*)?"
    r"(?:bis\s+\(\d+[a-z]?\)\s*)?(?:\(\s*)?"
    r"(?:weggefallen|aufgehoben|omitted|\.{3,})(?:\s*\))?\s*[.;]?\s*$"
)
MAIN_EDITORIAL_OMISSION_ONLY_SHA256 = (
    "c72cc7afe93f92aa86cb51bfeadc5a8a2995c011ac519443e18848b378387a1a"
)

HEBREW_BLOCKS = re.compile("[֐-׿יִ-ﭏ]")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- Literal alternatives, read from the compiled patterns --------------------


def _sequence(items) -> list[str]:
    spellings = [""]
    for op, value in items:
        options = _node(op, value)
        if options is None:
            return []
        spellings = [left + right for left in spellings for right in options]
    return spellings


def _node(op, value) -> list[str] | None:
    """The literal spellings of one parsed node; ``None`` if it needs more.

    A word-character run (``verbiet\\w*``) contributes its stem, a space class
    one space, an optional or bounded repetition its zero and one spellings,
    and a lookahead, which here only demands a following Hebrew letter, drops
    its alternative: no non-Hebrew text can satisfy it.
    """

    if op is sre_constants.LITERAL:
        return [chr(value)]
    if op is sre_constants.AT or op is sre_constants.ASSERT_NOT:
        return [""]
    if op is sre_constants.SUBPATTERN:
        return _sequence(list(value[3]))
    if op is sre_constants.BRANCH:
        return [spelling for branch in value[1] for spelling in _sequence(list(branch))]
    if op in (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT):
        low, _high, inner_items = value
        inner = _sequence(list(inner_items))
        if not inner:
            return [""] if low == 0 else None
        counts = (0, 1) if low == 0 else (low,)
        return [
            "".join(parts)
            for count in counts
            for parts in itertools.product(inner, repeat=count)
        ]
    if op is sre_constants.IN:
        kind, item = value[0]
        if kind is sre_constants.CATEGORY:
            return [" "] if item is sre_constants.CATEGORY_SPACE else [""]
        if kind is sre_constants.LITERAL:
            return [chr(item)]
        if kind is sre_constants.RANGE:
            return [chr(item[0])]
        return None
    return None


def pattern_literals(compiled: re.Pattern[str]) -> tuple[str, ...]:
    """Each top-level alternative's literal spellings, stripped and sorted."""

    tree = list(sre_parser.parse(compiled.pattern, compiled.flags))
    if len(tree) == 1 and tree[0][0] is sre_constants.BRANCH:
        alternatives = tree[0][1][1]
    else:
        alternatives = [tree]
    spellings = {
        spelling.strip()
        for alternative in alternatives
        for spelling in _sequence(list(alternative))
    }
    return tuple(sorted(spelling for spelling in spellings if spelling))


# The words the review and the main session named, whether or not a pattern
# lists them alone (``ohne`` stands before a computation word there).
LISTED_WORDS = (
    "kein",
    "nie",
    "weder",
    "statt",
    "anstatt",
    "ausser",
    "außer",
    "nicht",
    "niemals",
    "nirgends",
    "verboten",
    "ohne",
)


def _added_literals() -> frozenset[str]:
    """Every alternative this change added on a path English text once reached.

    ``_SOURCE_LINK_NEGATION_LANGUAGE`` read the raw scope at 82e71841a, and
    ``_NEGATION_LANGUAGE`` is new. An alternative main's English link guard
    already reads is main's, not an addition.
    """

    added = set()
    for compiled in (
        completeness._SOURCE_LINK_NEGATION_LANGUAGE,
        completeness._NEGATION_LANGUAGE,
    ):
        for literal in pattern_literals(compiled):
            if not completeness._ENGLISH_LINK_NEGATION_LANGUAGE.search(literal):
                added.add(literal)
    return frozenset(added)


INJECTED_WORDS = tuple(sorted(_added_literals() | set(LISTED_WORDS)))
NEUTRAL = "lee"


def case_spellings(word: str) -> tuple[str, ...]:
    """Lower, Title and upper case, each once."""

    return tuple(
        dict.fromkeys((word.lower(), word[:1].upper() + word[1:], word.upper()))
    )


# --- Frames ------------------------------------------------------------------

ENGLISH_CITATION = "us-nj/statute/54a:8-6"
ENGLISH_OUTPUT = "us-nj:statutes/54a:8-6/a#deferred_output"
ENGLISH_REASON = "Missing section 2."
ENGLISH_SOURCE = "(a) It is paid in accordance with section 2."
GERMAN_CITATION = "de/statute/estg/32a"
GERMAN_OUTPUT = "de:statutes/estg/32a/1#deferred_output"
GERMAN_REASON = "Fehlt EStG § 26."
GERMAN_SOURCE = "(1) Maßgeblich ist die Bemessungsgrundlage nach § 26."

ENGLISH_SOURCE_FRAMES = (
    "(a) Tax is computed for {w} in accordance with section 2.",
    "(a) The amount is calculated for {w} in accordance with section 2.",
    "(a) The amount is calculated for Ms {w} in accordance with section 2.",
    "(a) The credit of {w} is determined in accordance with section 2.",
    "(a) {w} shall pay the tax in accordance with section 2.",
    "(a) The benefit is paid to {w} in accordance with section 2.",
    "(a) The liability of {w} arises in accordance with section 2.",
    "(a) The tax imposed on {w} is computed under section 2.",
    "(a) The allowance for {w} is computed pursuant to section 2.",
    "(a) The amount, as {w} reported, is calculated in accordance with section 2.",
    # Refused frames: the issue strings must match too.
    "(a) Tax is computed for {w} in accordance with section 3.",
    "(a) Tax is computed for {w}.\n(b) It is paid in accordance with section 2.",
)
ENGLISH_REASON_FRAMES = (
    "Missing section 2 for {w}.",
    "{w}: missing section 2.",
    "Missing section 2, as {w} noted.",
    "The computation for {w} is missing section 2.",
    "Missing section 2, which {w} cites.",
    "Section 2 is missing for {w}.",
    "Missing section 2 (per {w}).",
    "Missing section 2; {w} agrees.",
    "Missing section 2 as applied to {w}.",
    "Cannot be computed for {w} until section 2 is encoded.",
)
GERMAN_SOURCE_FRAMES = (
    "(1) Maßgeblich ist für {w} die Bemessungsgrundlage nach § 26.",
    "(1) Die Steuer für {w} bemisst sich nach § 26.",
    "(1) {w} zahlt die Steuer nach § 26.",
    "(1) Der Betrag für Frau {w} richtet sich nach § 26.",
    "(1) Maßgeblich ist die Bemessungsgrundlage von {w} gemäß § 26.",
    "(1) Die Zulage für {w} wird gemäß § 26 berechnet.",
)
GERMAN_REASON_FRAMES = (
    "Fehlt EStG § 26 für {w}.",
    "{w}: Fehlt EStG § 26.",
    "Fehlt EStG § 26, wie {w} meint.",
    "Für {w} fehlt EStG § 26.",
    "Fehlt EStG § 26 (laut {w}).",
)
FRAMES = {
    "en-source": ENGLISH_SOURCE_FRAMES,
    "en-reason": ENGLISH_REASON_FRAMES,
    "de-source": GERMAN_SOURCE_FRAMES,
    "de-reason": GERMAN_REASON_FRAMES,
}


def frame_case(kind: str, frame: str, spelling: str) -> dict[str, str]:
    filled = frame.format(w=spelling)
    if kind.startswith("en"):
        case = {
            "citation": ENGLISH_CITATION,
            "output": ENGLISH_OUTPUT,
            "profile": "en-US",
            "reason": ENGLISH_REASON,
            "source": ENGLISH_SOURCE,
        }
    else:
        case = {
            "citation": GERMAN_CITATION,
            "output": GERMAN_OUTPUT,
            "profile": "de-DE",
            "reason": GERMAN_REASON,
            "source": GERMAN_SOURCE,
        }
    case["source" if kind.endswith("source") else "reason"] = filled
    return case


def _quoted(reason: str) -> str:
    quoted = json.dumps(reason, ensure_ascii=False)
    return "".join(
        f"\\u{ord(char):04x}" if unicodedata.category(char) == "Cc" else char
        for char in quoted
    )


def _content(citation: str, output: str, reason: str) -> str:
    return (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {citation}\n"
        "  deferred_outputs:\n"
        f"    - output: {output}\n"
        f"      reason: {_quoted(reason)}\n"
        "rules: []\n"
    )


@functools.lru_cache(maxsize=None)
def _issues(
    citation: str, content: str, source: str, profile: str = "en-US"
) -> tuple[str, ...]:
    return analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues


def _case_issues(case: dict[str, str]) -> tuple[str, ...]:
    content = case.get("content") or _content(
        case["citation"], case["output"], case["reason"]
    )
    return _issues(
        case["citation"], content, case["source"], case.get("profile", "en-US")
    )


def _verdict(issues: tuple[str, ...]) -> bool:
    return not any(issue.startswith((DEFERRAL, STRUCTURE)) for issue in issues)


def neutral_spelling(spelling: str) -> str:
    """``Lee`` stretched to the spelling's shape: ``abgesehen von`` -> ``leeeeeeee lee``.

    Each letter run becomes ``L`` and then ``e``s, in the spelling's case, and
    every other character stays, so offsets and word counts echoed in issue
    strings match the injected frame's.
    """

    shaped = []
    for index, char in enumerate(spelling):
        if not char.isalpha():
            shaped.append(char)
            continue
        first = index == 0 or not spelling[index - 1].isalpha()
        letter = "l" if first else "e"
        shaped.append(letter.upper() if char.isupper() else letter)
    return "".join(shaped)


@functools.lru_cache(maxsize=None)
def _main_reads() -> dict[tuple[str, str, str], tuple[str, ...]]:
    """Main's issue arrays where main itself reads an injected word.

    ``ausgenommen``, ``außer`` and ``außerstande`` are source-stated controls
    to main's formula-output reader, and ``no`` negates a missing signal; there
    the neutral name cannot stand in for main, so main's exact array does.
    """

    pinned = json.loads(MAIN_ISSUES.read_text())
    return {
        (row["kind"], row["frame"], row["spelling"]): tuple(row["issues"])
        for row in pinned["main_reads"]
    }


def _assert_same_as_neutral(kind: str, frame: str, spelling: str) -> None:
    neutral = neutral_spelling(spelling)
    case = frame_case(kind, frame, spelling)
    assert not HEBREW_BLOCKS.search(case["reason"] + case["source"])
    assert not hebrew_source_text.holds_hebrew_script(case["reason"], case["source"])
    issues = _case_issues(case)
    main_issues = _main_reads().get((kind, frame, spelling))
    if main_issues is not None:
        assert issues == main_issues, (kind, frame, spelling)
        return
    neutral_issues = _case_issues(frame_case(kind, frame, neutral))
    # An issue that echoes the text names the injected word where the neutral
    # frame names its ``Lee``; nothing else may differ.
    expected = tuple(issue.replace(neutral, spelling) for issue in neutral_issues)
    assert (_verdict(issues), issues) == (_verdict(neutral_issues), expected), (
        kind,
        frame,
        spelling,
    )


# --- The English link guard ----------------------------------------------------


def test_english_link_negations_are_mains():
    pattern = completeness._ENGLISH_LINK_NEGATION_LANGUAGE
    assert _sha256(pattern.pattern) == MAIN_ENGLISH_LINK_NEGATION_SHA256
    assert pattern.flags == re.IGNORECASE | re.UNICODE


def test_a_raw_scope_reads_only_the_english_link_negations(monkeypatch):
    """Only the cross-language copy of a Hebrew scope reads the German markers."""

    read: list[str] = []
    wide = completeness._SOURCE_LINK_NEGATION_LANGUAGE

    class Recording:
        pattern = wide.pattern
        flags = wide.flags

        @staticmethod
        def finditer(text, *args):
            read.append(text)
            return wide.finditer(text, *args)

    monkeypatch.setattr(completeness, "_SOURCE_LINK_NEGATION_LANGUAGE", Recording)
    for scope in (
        "(a) Tax is computed for Nie in accordance with",
        "(a) Steuer wird nicht in accordance with",
        "(a) The amount is not calculated in accordance with",
    ):
        completeness._source_link_scope_has_unreset_negation(scope)
    assert read == []
    completeness._source_link_scope_has_unreset_negation(
        "(א) Tax is computed for Nie in accordance with"
    )
    assert read and all(HEBREW_BLOCKS.search(text) for text in read)


@pytest.mark.parametrize(
    ("scope", "negated"),
    (
        ("(a) Tax is computed for Nie in accordance with", False),
        ("(a) Tax is computed for Kein in accordance with", False),
        ("(a) Tax is nicht computed in accordance with", False),
        ("(a) Tax is not computed in accordance with", True),
        # A Hebrew scope reads the German markers in its cross-language copy.
        ("(א) Tax is nicht computed in accordance with", True),
        ("(א) Tax is computed for Lee in accordance with", False),
    ),
)
def test_scope_negation_by_script(scope, negated):
    assert completeness._source_link_scope_has_unreset_negation(scope) is negated


# --- Shared English patterns that gained Hebrew alternatives -------------------


@pytest.mark.parametrize("name", sorted(MAIN_SHARED_PATTERNS))
def test_shared_pattern_is_mains_plus_a_hebrew_suffix(name):
    main_pattern, digest, hebrew_suffix = MAIN_SHARED_PATTERNS[name]
    compiled = getattr(completeness, name)
    assert _sha256(main_pattern) == digest
    assert compiled.pattern == f"{main_pattern}|{hebrew_suffix}"
    assert compiled.flags == re.IGNORECASE | re.UNICODE


def test_editorial_omission_pattern_is_mains_on_non_hebrew_text():
    assert _sha256(MAIN_EDITORIAL_OMISSION_ONLY) == MAIN_EDITORIAL_OMISSION_ONLY_SHA256
    main = re.compile(MAIN_EDITORIAL_OMISSION_ONLY, flags=re.IGNORECASE)
    for text in (
        "(1) weggefallen",
        "(a) (aufgehoben)",
        "2a. omitted.",
        "...",
        "(b) Omitted;",
        "bis (3) weggefallen",
        "(1) gilt nicht",
        "(a) It is paid.",
    ):
        assert bool(completeness._EDITORIAL_OMISSION_ONLY.match(text)) is bool(
            main.match(text)
        ), text


_PATTERN_ALPHABET = st.sampled_from(
    [
        *"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüßÄÖÜ0123456789",
        *" \t\n.,;:()-'\"’§",
        "‏",
        "­",
        "missing ",
        "requires ",
        "not yet encoded ",
        "without ",
        "fehlt ",
        "nicht codiert ",
        "although ",
        "but ",
        "yet ",
        "aber ",
        "form ",
        "data ",
        "(1) weggefallen",
        "omitted",
        "aufgehoben",
        *(f"{word} " for word in LISTED_WORDS),
    ]
)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(st.lists(_PATTERN_ALPHABET, max_size=40).map("".join))
def test_shared_patterns_read_non_hebrew_text_as_main_does(text):
    assert not HEBREW_BLOCKS.search(text)
    for name, (main_pattern, _digest, _suffix) in MAIN_SHARED_PATTERNS.items():
        main = re.compile(main_pattern, flags=re.IGNORECASE)
        shared = getattr(completeness, name)
        assert [match.span() for match in shared.finditer(text)] == [
            match.span() for match in main.finditer(text)
        ], name
    main_omission = re.compile(MAIN_EDITORIAL_OMISSION_ONLY, flags=re.IGNORECASE)
    assert bool(completeness._EDITORIAL_OMISSION_ONLY.match(text)) is bool(
        main_omission.match(text)
    )
    assert hebrew_source_text.fold_marks(text) is text
    assert hebrew_source_text.cross_language_text(text) is None
    assert hebrew_source_text.english_reading_view(text) is None
    assert not hebrew_source_text.denies_linked_claim(text, text)
    assert not hebrew_source_text.ends_with_negation(text)
    assert not hebrew_source_text.states_explicit_computation(text)
    assert hebrew_source_text.hebrew_outline_markers(text) == ()
    assert hebrew_source_text.strip_leading_outline_chain(text) is None


# --- Word injection ------------------------------------------------------------


def test_injected_words_cover_the_listed_and_added_alternatives():
    assert set(LISTED_WORDS) <= set(INJECTED_WORDS)
    assert _added_literals() <= set(INJECTED_WORDS)
    # The German link words of finding 5 and the new negation pattern's.
    for word in ("nie", "kein", "keine", "keiner", "weder", "anstatt", "niemals"):
        assert word in INJECTED_WORDS
    for word in INJECTED_WORDS:
        assert not HEBREW_BLOCKS.search(word), word
    assert len(FRAMES["en-source"]) + len(FRAMES["en-reason"]) >= 20
    assert len(FRAMES["de-source"]) + len(FRAMES["de-reason"]) >= 10


@pytest.mark.parametrize("word", INJECTED_WORDS)
def test_an_injected_word_changes_nothing_a_neutral_name_does_not(word):
    for spelling in case_spellings(word):
        for kind, frames in FRAMES.items():
            for frame in frames:
                _assert_same_as_neutral(kind, frame, spelling)


def test_the_neutral_frames_accept_where_they_should():
    """The injection compares against frames that can flip, not only refusals."""

    accepted = {
        kind: [
            _verdict(_case_issues(frame_case(kind, frame, "Lee"))) for frame in frames
        ]
        for kind, frames in FRAMES.items()
    }
    assert all(accepted["en-source"][:10])
    assert not any(accepted["en-source"][10:])
    assert sum(accepted["en-reason"]) >= 5
    assert all(accepted["de-source"])
    assert sum(accepted["de-reason"]) >= 3


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    st.sampled_from(INJECTED_WORDS),
    st.sampled_from(sorted(FRAMES)),
    st.integers(min_value=0, max_value=11),
    st.sampled_from(("lower", "title", "upper")),
    st.lists(
        st.sampled_from(("tax", "Ms", "the", "Dr.", "Lee", "und", "for", "of")),
        max_size=3,
    ),
)
def test_a_word_swapped_for_lee_never_changes_a_non_hebrew_verdict(
    word, kind, index, case, filler
):
    frames = FRAMES[kind]
    frame = frames[index % len(frames)]
    frame = frame.replace("{w}", " ".join((*filler, "{w}")))
    spelling = {
        "lower": word.lower(),
        "title": word[:1].upper() + word[1:],
        "upper": word.upper(),
    }[case]
    if filler and word in {row[2].lower() for row in _main_reads()}:
        # Main reads this word in some frames; its pinned arrays cover only the
        # listed frames, which the exhaustive injection test checks.
        return
    _assert_same_as_neutral(kind, frame, spelling)


def test_main_reads_only_the_pinned_words():
    assert {spelling.lower() for _kind, _frame, spelling in _main_reads()} == {
        "ausgenommen",
        "außer",
        "außerstande",
        "no",
    }


# --- Main's exact issue arrays ---------------------------------------------------


def _main_cases() -> list[dict]:
    pinned = json.loads(MAIN_ISSUES.read_text())
    assert pinned["main"] == MAIN
    return pinned["cases"]


@pytest.mark.parametrize("case", _main_cases(), ids=lambda case: case["id"])
def test_pinned_non_hebrew_cases_keep_mains_issue_arrays(case):
    assert not hebrew_source_text.holds_hebrew_script(
        case.get("reason", ""), case["source"], case.get("content", "")
    )
    assert list(_case_issues(case)) == case["issues"]


def test_review_finding_6_keeps_mains_source_unit_diagnostic():
    (case,) = [case for case in _main_cases() if case["id"] == "f6"]
    issue = case["issues"][0]
    assert issue.startswith(
        "[complete-source-unit:deferral] `module.deferred_outputs[0]` identifies "
        "source branch (source unit) (`<source-unit>`) but its deferral does not "
        "name an exact missing dependency"
    )
    assert "cites exact source branch" not in issue
    assert "gematria" not in issue


# --- A rule's Hebrew `source:` annotation (round 22) ------------------------------
#
# The native citation helpers read a third input, a rule's own `source:`. They run
# only behind the wall's gate: the candidate's source or a deferral reason holds
# Hebrew. A Hebrew annotation over an English reason and source takes main's array.


def _annotation_cases() -> list[dict]:
    pinned = json.loads(MAIN_ISSUES.read_text())
    assert pinned["main"] == MAIN
    return pinned["annotation_cases"]


def _annotation_controls() -> list[dict]:
    pinned = json.loads(MAIN_ISSUES.read_text())["annotation_controls"]
    assert pinned["pinned_at"] == "5dcf2bc13132673056dbd9f52a2b8794f0e6fe7b"
    return pinned["cases"]


@pytest.mark.parametrize("case", _annotation_cases(), ids=lambda case: case["id"])
def test_a_hebrew_annotation_beside_english_text_keeps_mains_issue_array(case):
    assert not hebrew_source_text.holds_hebrew_script(case["reason"], case["source"])
    assert list(_case_issues(case)) == case["issues"]


def test_the_annotation_cases_cover_the_review_frames():
    by_id = {case["id"]: case for case in _annotation_cases()}
    # Sol's minimized case and its `Nie`/`Lee` variants refuse on main; the
    # canonical annotation accepts. A word and `Lee` agreeing is not enough.
    for who in ("", ":Nie", ":Lee"):
        assert (
            'source: "סעיף 1(א)"' in by_id[f"en-source:hebrew-branch{who}"]["content"]
        )
        assert not _verdict(tuple(by_id[f"en-source:hebrew-branch{who}"]["issues"]))
        assert _verdict(tuple(by_id[f"en-source:canonical{who}"]["issues"]))
        assert not _verdict(tuple(by_id[f"en-source:unrecognized{who}"]["issues"]))


@pytest.mark.parametrize("case", _annotation_controls(), ids=lambda case: case["id"])
def test_a_hebrew_annotation_behind_the_gate_keeps_its_reading(case):
    assert hebrew_source_text.holds_hebrew_script(case["reason"], case["source"])
    assert list(_case_issues(case)) == case["issues"]


def test_a_hebrew_annotation_over_a_hebrew_source_still_covers_its_branch():
    (case,) = [
        case
        for case in _annotation_controls()
        if case["id"] == "he-source:hebrew-branch"
    ]
    assert case["issues"] == []
