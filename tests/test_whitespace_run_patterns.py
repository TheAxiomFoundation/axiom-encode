r"""Rewritten whitespace patterns match as the patterns they replaced did, and
stay linear in a run of spaces.

Each replaced pattern put two whitespace runs side by side around an optional
item, as ``\s*(?:usd)?\s*`` does. A run of n spaces then splits n + 1 ways, and
text that fails after the run backtracks through every split: quadratic in the
run, or cubic where two such pairs follow each other. Each rewrite hangs the
trailing whitespace on the optional item, ``\s*(?:usd\s*)?``, which accepts the
same strings with one way through the run.

Each case keeps its replaced pattern verbatim. The differential property
compares every match, span and group of the two on generated text, so it fails
a rewrite that changes anything a caller reads.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness import validator_pipeline as pipeline
from tests.scaling import assert_linear_growth, lowest_work

_CARDINAL = completeness._ENGLISH_CARDINAL_PHRASE

_SPACE = st.sampled_from(
    ["", " ", "  ", "     ", "\t", "\n", " \t ", "\xa0", " ", "　"]
)
_GAP = st.lists(_SPACE, max_size=3).map("".join)
_TAIL = st.sampled_from(["", "", "x", " x", ")", "1", "(", ":", "\n", " ."])
_SPELLED = st.sampled_from(
    [
        "five",
        "Twenty five",
        "twenty-five",
        "one hundred and five",
        "five hundred  thousand",
        "fivee",
    ]
)
_UNIT = st.sampled_from(
    ["dollars", "Dollar", "usd", "euros", "EUR", "pounds", "gbp", "yen", ""]
)
_MARKER = st.sampled_from(["", "", "$", "€", "£", "USD", "eur", "gbp", "x"])
_TOKEN_SOUP = st.lists(
    st.sampled_from(
        [
            " ",
            "  ",
            "\t",
            "\n",
            "\xa0",
            "(",
            ")",
            "$",
            "€",
            ",",
            ".",
            ":",
            "|",
            "=>",
            "-",
            "*",
            "usd",
            "eur",
            "dollars",
            "five",
            "and",
            "an",
            "x",
            "1",
            "1,000",
            "2024",
            "ARTICLE",
            "step",
            "above",
            "size",
            "first",
            "may",
            "von",
        ]
    ),
    max_size=12,
).map("".join)


def _text(*parts: str | list[str] | st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    """Join parts in order; a list is one choice among its strings."""

    return st.tuples(
        *(
            st.just(part)
            if isinstance(part, str)
            else st.sampled_from(part)
            if isinstance(part, list)
            else part
            for part in parts
        )
    ).map("".join)


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    current: re.Pattern[str]
    # The pattern this change replaced, verbatim.
    replaced: str
    # How every caller applies the pattern.
    mode: str
    near_matches: st.SearchStrategy[str]
    # Texts at the edges the rewrite moved: the optional item present or
    # absent, with whitespace beside it or none. The first one matches.
    boundaries: tuple[str, ...]
    # A run of ``width`` spaces that the pattern then fails on.
    adversarial: Callable[[int], str]


CASES = (
    Case(
        "parenthetical restatement gap",
        completeness._PARENTHETICAL_RESTATEMENT_GAP,
        r"\s*(?:dollars?|usd|euros?|eur|pounds?|gbp)?\s*"
        r"\(\s*(?:\$|€|£|usd|eur|gbp)?\s*",
        "fullmatch",
        _text(_GAP, _UNIT, _GAP, ["(", "", "(("], _GAP, _MARKER, _GAP, _TAIL),
        (
            " dollars ( $ ",
            "(",
            " (",
            "dollars(",
            "dollars (",
            " ( $",
            "($",
            "( usd ",
            "(usd",
            "dollars",
            "dollars ( $ x",
        ),
        lambda width: " " * width + "(" + " " * width + "x",
    ),
    Case(
        "spelled parenthetical first gap",
        completeness._SPELLED_PARENTHETICAL_FIRST_GAP,
        rf"\s*{_CARDINAL}\s+"
        r"(?:dollars?|usd|euros?|eur|pounds?|gbp)\s*"
        r"\(\s*(?:\$|€|£|usd|eur|gbp)?\s*",
        "fullmatch",
        _text(_GAP, _SPELLED, _GAP, _UNIT, _GAP, ["(", ""], _GAP, _MARKER, _GAP, _TAIL),
        (
            " five dollars ($",
            "five dollars(",
            "five dollars ( ",
            " five dollars ( $ ",
            "five dollars ($ ",
            "five dollars ( usd",
            "five dollars",
            "five  dollars",
        ),
        lambda width: " five dollars (" + " " * width + "x",
    ),
    Case(
        "first gap filler",
        completeness._FIRST_GAP_FILLER,
        r"\s*(?:\$|€|£|usd|eur|gbp)?\s*(?:(?:zu|bis)\s+)?"
        r"(?:(?:einschließlich|maximal|inklusive|including|maximum)\s+)?"
        r"(?:(?:a|an|the|einem?|einer|dem|der|das)\s+)?"
        r"(?:(?:zu\s+versteuernd\w*|maßgeblich\w*)\s+)?"
        r"(?:(?:einkommen|betrag|wert|income|amount|age)\s+)?"
        r"(?:(?:von|of)\s+)?"
        r"(?:(?:einschließlich|maximal|inklusive|including|maximum)\s+)?",
        "fullmatch",
        _text(
            _GAP,
            _MARKER,
            _GAP,
            ["", "zu ", "bis\t"],
            ["", "einschließlich ", "including  "],
            ["", "the ", "a", "dem "],
            ["", "zu versteuernde ", "maßgeblichen "],
            ["", "income ", "Einkommen\n", "amount"],
            ["", "of ", "von "],
            ["", "maximum ", "inklusive "],
            _GAP,
            _TAIL,
        ),
        (
            " $ including the income of ",
            "",
            " ",
            "$",
            "$ ",
            " $",
            "$income ",
            "usd of ",
            " $ x",
        ),
        lambda width: " " * width + "x",
    ),
    Case(
        "von open range continuation",
        completeness._VON_OPEN_RANGE_CONTINUATION,
        r"\s*(?:[^\W\d_]+(?:-[^\W\d_]+)?|[%€$£]|v\.\s*H\.)?\s+"
        r"(?:an|aufwärts)\b",
        "match",
        _text(
            _GAP,
            ["", "Euro", "v. H.", "v.H.", "%", "€", "Euro-Cent", "x2", "an"],
            _GAP,
            ["an", "aufwärts", "ab", "", "anx"],
            _TAIL,
        ),
        (
            " Euro an",
            "Euro an",
            " an",
            "an",
            "  an",
            " v. H. an",
            "%an",
            "% an",
            " Euro",
            " Euro x",
        ),
        lambda width: " " * width + "x",
    ),
    Case(
        "spelled parenthetical amount",
        completeness._SPELLED_PARENTHETICAL_AMOUNT,
        rf"\b(?P<words>{_CARDINAL})\s+"
        r"(?P<unit>dollars?|usd|euros?|eur|pounds?|gbp)\s*"
        r"\(\s*(?P<marker>\$|€|£|usd|eur|gbp)?\s*"
        r"(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
        r"\s*\)",
        "finditer",
        _text(
            ["", "pay ", "x"],
            _SPELLED,
            _GAP,
            _UNIT,
            _GAP,
            ["(", ""],
            _GAP,
            _MARKER,
            _GAP,
            ["5", "1,000", "1,000.50", "12.5", "1,00", "x"],
            _GAP,
            [")", "", "x)"],
            _TAIL,
        ),
        (
            "five dollars ($5)",
            "five dollars($5)",
            "five dollars ( $ 5 )",
            "five dollars (5)",
            "five dollars (usd5)",
            "five dollars ($ x)",
            "five dollars ( 5",
        ),
        lambda width: "five dollars (" + " " * width + "x",
    ),
    Case(
        "trailing spelled parenthetical opening",
        completeness._TRAILING_SPELLED_PARENTHETICAL_OPENING,
        rf"\b{_CARDINAL}\s+"
        r"(?:dollars?|usd|euros?|eur|pounds?|gbp)\s*"
        r"\(\s*(?:\$|€|£|usd|eur|gbp)?\s*$",
        "finditer",
        _text(
            ["", "at least ", "x"],
            _SPELLED,
            _GAP,
            _UNIT,
            _GAP,
            ["(", ""],
            _GAP,
            _MARKER,
            _GAP,
            ["", "", "x", "\n", "1"],
        ),
        (
            "at least five dollars ($",
            "five dollars(",
            "five dollars ( ",
            "five dollars ( $ ",
            "five dollars ($x",
            "five dollars",
        ),
        lambda width: "five dollars (" + " " * width + "x",
    ),
    Case(
        "structural source heading",
        pipeline._STRUCTURAL_SOURCE_HEADING_PATTERN,
        r"^(?:(?:PART|CHAPTER|SCHEDULE|REGULATION)\b|"
        r"ARTICLE\b\s*(?:\d+[A-Za-z]?)?\s*$)",
        "match",
        _text(
            ["ARTICLE", "Article", "PART", "chapter", "ARTICLES", "x"],
            _GAP,
            ["", "1", "12a", "3B", "iv"],
            _GAP,
            _TAIL,
        ),
        (
            "ARTICLE 12a",
            "ARTICLE",
            "ARTICLE ",
            "ARTICLE12",
            "ARTICLE 1 ",
            "ARTICLE 1 x",
            "ARTICLE x",
        ),
        lambda width: "ARTICLE" + " " * width + "x",
    ),
    Case(
        "step reference",
        pipeline._SOURCE_REFERENCE_PATTERNS[2],
        rf"\b(?:step|steps)\s+{pipeline._SOURCE_REFERENCE_SEQUENCE_PATTERN}"
        r"(?:\s*,?\s*(?:above|below))?",
        "finditer",
        _text(
            ["", "see "],
            ["step", "steps", "Step"],
            [" ", "  ", "\t"],
            ["1", "2b", "(a)", "iv", "A", "3(b)", "4.5"],
            ["", ", 2", " and 3", ", and (c)"],
            _GAP,
            ["", ","],
            _GAP,
            ["", "above", "below", "abov"],
            _TAIL,
        ),
        (
            "step 2, above",
            "step 2,above",
            "step 2 above",
            "step 2above",
            "step 2 , above",
            "step 2,",
            "step 2 x",
        ),
        lambda width: "step 1" + " " * width + "x",
    ),
    Case(
        "schedule size row",
        pipeline._SCHEDULE_SIZE_ROW_PATTERN,
        r"^\s*[-*]?\s*(?:size|household size|unit size)"
        r"(?:\s+\d+(?:\s+or\s+more)?)?\s*:\s*"
        r"(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
        "fullmatch",
        _text(
            _GAP,
            ["", "-", "*", "--"],
            _GAP,
            ["size", "household size", "unit size", "sizes"],
            ["", " 1", " 2 or more", "\t3"],
            _GAP,
            [":", ""],
            _GAP,
            ["", "$", "£", "€"],
            _GAP,
            ["100", "-5", "1,234.50", "x"],
            _GAP,
            ["", "", "x"],
        ),
        (
            "- size 2: $100",
            "size:100",
            "-size:100",
            " - size : 100",
            "*size 1: $ 5",
            "- size",
            "-: 100",
        ),
        lambda width: " " * width + "x",
    ),
    Case(
        "schedule pipe row",
        pipeline._SCHEDULE_PIPE_ROW_PATTERN,
        r"^\s*[-*]?\s*(?:\d+(?:\s+or\s+more)?|size\s+\d+(?:\s+or\s+more)?|"
        r"household size\s+\d+(?:\s+or\s+more)?|unit size\s+\d+(?:\s+or\s+more)?)\s*\|\s*"
        r"(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
        "fullmatch",
        _text(
            _GAP,
            ["", "-", "*"],
            _GAP,
            ["1", "2 or more", "size 3", "household size 2", "unit size 1", "x"],
            _GAP,
            ["|", ""],
            _GAP,
            ["", "$"],
            _GAP,
            ["100", "1,000.5"],
            _GAP,
            ["", "x"],
        ),
        (
            "* 2 or more | $100",
            "1|100",
            "-1|100",
            " - 1 | 100",
            "* | 100",
            "- x | 100",
        ),
        lambda width: " " * width + "x",
    ),
    Case(
        "schedule arrow row",
        pipeline._SCHEDULE_ARROW_ROW_PATTERN,
        r"^\s*[-*]?\s*(?:\d+(?:\s+or\s+more)?|size\s+\d+(?:\s+or\s+more)?|"
        r"household size\s+\d+(?:\s+or\s+more)?|unit size\s+\d+(?:\s+or\s+more)?)\s*"
        r"(?:=>|->|=)\s*(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
        "fullmatch",
        _text(
            _GAP,
            ["", "-", "*"],
            _GAP,
            ["1", "2 or more", "size 3", "household size 2", "x"],
            _GAP,
            ["=>", "->", "=", ""],
            _GAP,
            ["", "$"],
            _GAP,
            ["100", "1,000.5"],
            _GAP,
            ["", "x"],
        ),
        (
            "size 3 => 1,000",
            "1=>100",
            "-1=>100",
            " - 1 -> 100",
            "* 1 = 100",
            "- => 100",
        ),
        lambda width: " " * width + "x",
    ),
    Case(
        "two-line table row key",
        pipeline._TWO_LINE_TABLE_ROW_KEY_PATTERN,
        r"^\s*(?:\d{1,3}|"
        r"(?:(?:\d{1,3}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|"
        r"seventh|eighth|ninth|tenth)\s+[A-Za-z][A-Za-z -]*))\s*$",
        "fullmatch",
        _text(
            _GAP,
            ["1", "12", "1st", "first", "Tenth", "2nd", "1234"],
            ["", " ", "  ", "\t"],
            ["", "a", "Person", "x-y", "two words", "a -", "b  c", "-"],
            _GAP,
            ["", "", "1", "x"],
        ),
        (
            "1st person",
            "first a",
            "first a ",
            "first a -",
            "first a - ",
            "first a-",
            "first -",
            "first  a b  ",
            "1",
        ),
        lambda width: "first a" + " " * width + "1",
    ),
    Case(
        "value-bearing table row",
        pipeline._VALUE_BEARING_TABLE_ROW_PATTERN,
        r"^\s*[-*]?\s*[^:]+:\s*(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
        "match",
        _text(
            _GAP,
            ["", "-", "*"],
            _GAP,
            ["Amount", "a b", "", " ", "\t", "x-y", "-"],
            [":", "", "::"],
            _GAP,
            ["", "$", "€"],
            _GAP,
            ["100", "-1,000.25", "x"],
            _GAP,
            ["", "x"],
        ),
        (
            "  - Amount: $1,000.50",
            ":100",
            " :100",
            "\t:100",
            "-:100",
            "- :100",
            " - :100",
            "*:5",
            "- Amount:x",
        ),
        lambda width: " " * width + "x",
    ),
    Case(
        "English month day year",
        pipeline._ENGLISH_MONTH_DAY_YEAR_PATTERN,
        rf"\b{pipeline._MONTH_NAME_BODY}\s+(?P<day>0?[1-9]|[12]\d|3[01])"
        rf"(?:st|nd|rd|th)?\s*,?\s*(?P<year>{pipeline._TEMPORAL_YEAR_BODY})"
        rf"{pipeline._TEMPORAL_YEAR_END}",
        "finditer",
        _text(
            ["", "on "],
            ["may", "May", "jan", "January", "sept", "Sept."],
            [" ", "  ", "\t", ""],
            ["1", "01", "31", "32", "9"],
            ["", "st", "th"],
            _GAP,
            ["", ","],
            _GAP,
            ["2024", "1999", "1850", "2100", "x"],
            ["", ".", "x", ",5", " "],
        ),
        (
            "on May 1, 2024",
            "may 1,2024",
            "may 1 ,2024",
            "may 1 , 2024",
            "may 12024",
            "may 1st, 2024",
            "may 1 x",
        ),
        lambda width: "may 1" + " " * width + "x",
    ),
)
_IDS = [case.name for case in CASES]


def _observe(pattern: re.Pattern[str], mode: str, text: str) -> object:
    """Everything a caller can read from applying ``pattern`` the case's way."""

    def details(match: re.Match[str] | None) -> object:
        if match is None:
            return None
        return (
            match.span(),
            match.groups(),
            tuple(match.span(group) for group in range(1, pattern.groups + 1)),
        )

    if mode == "finditer":
        return [details(match) for match in pattern.finditer(text)]
    return details(getattr(pattern, mode)(text))


# Main compiled every replaced pattern with these flags.
_REPLACED_FLAGS = re.IGNORECASE


def _replaced(case: Case) -> re.Pattern[str]:
    return re.compile(case.replaced, _REPLACED_FLAGS)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_rewritten_pattern_keeps_the_replaced_flags_and_groups(case: Case):
    replaced = _replaced(case)

    assert case.current.flags == replaced.flags
    assert case.current.groupindex == replaced.groupindex
    assert case.current.groups == replaced.groups


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_rewritten_pattern_reads_each_boundary_as_the_replaced_one(case: Case):
    replaced = _replaced(case)

    assert _observe(case.current, case.mode, case.boundaries[0]) not in (None, [])
    for text in (*case.boundaries, case.adversarial(64)):
        assert _observe(case.current, case.mode, text) == _observe(
            replaced, case.mode, text
        ), text


@pytest.mark.parametrize("case", CASES, ids=_IDS)
@settings(max_examples=400, deadline=None, derandomize=True)
@given(data=st.data())
def test_rewritten_pattern_reads_every_match_as_the_replaced_one(case: Case, data):
    text = data.draw(st.one_of(case.near_matches, _TOKEN_SOUP), label="text")

    assert _observe(case.current, case.mode, text) == _observe(
        _replaced(case), case.mode, text
    )


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_rewritten_pattern_stays_linear_in_a_space_run(case: Case):
    def work(width):
        text = case.adversarial(width)
        expected = _observe(case.current, case.mode, text)

        def check(observed):
            assert observed == [expected] * 4

        return lowest_work(
            lambda: [_observe(case.current, case.mode, text) for _ in range(4)],
            check,
        )

    # The first step fails each replaced pattern, quadratic or cubic, within
    # seconds, so a regression never reaches the wide pair, which would run a
    # cubic one for hours. The wide pair holds the patterns to the run widths
    # the interval parser's own growth test uses.
    assert_linear_growth(work(64), work(512), input_ratio=8)
    assert_linear_growth(work(1_000), work(16_000), input_ratio=16)
