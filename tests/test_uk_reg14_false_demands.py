"""Four false refusals of regulation 14 in complete-source analysis.

Signed re-encode run 41745442025 ended with refusals of regulation 14 of the
Council Tax Reduction (Scotland) Regulations 2012 that came from three misreadings
of its text. Each one recurs in every re-encode of the regulation, and council tax
reduction schemes across the UK repeat the same wording. The gate read:

* "amount B is 2 6/7 per cent of the difference" as a division, because of the
  slash, and as a computation over the 2, 6 and 7 inside the rate;
* "falls", which German uses for "if", as a condition in "the day falls within a
  week";
* condition words inside the title of another provision as conditions of this
  one, as in "regulation 57 (residents of a dwelling to whom regulation 14(6)
  does not apply)".

A fourth false refusal waited behind those three. The run's candidate stated
amount B as the integer 1, and the gate computes 0.99999999999999995. The gate
accepted that residue only against a decimal 1.0, so a model that tested every
condition and stated amount B as 1 still drew a demand for paragraph (8)(b)'s
case pair.

The tests below pin these invariants:

* outside corpora under de/ and dk/, a digit mixed percentage written with "/"
  yields one rate, (whole + numerator / denominator) / 100, and its parts yield
  nothing;
* its slash adds no division and no computation to operation kinds, stated
  computations, rounding-only clauses or colon headings, and a real division in
  the same text still counts;
* the collapse in _formula_branch_computation_occurrences removes only
  occurrences nested in a mixed number the extractor reads whole: a mixed
  percentage, or a mixed number written with the fraction slash;
* outside corpora under de/ and dk/, the slash mask
  (_without_digit_mixed_percentage_slashes) and the collapse agree on a numeral
  written with "/": where the mask hides the slash the gate keeps none of the
  numeral's parts, and where it leaves the slash the collapse drops none;
* on one line, after a space and with a whole number of digits only, the mask
  hides an unspaced slash, as in "2 6/7", if and only if the extractor reads the
  whole numeral as a rate of its mixed value. The collapse matches a mixed number
  written with the fraction slash (U+2044), with or without spaces around the
  slash, if and only if the extractor reads it whole as a mixed number. It
  matches one written with a vulgar fraction if and only if the extractor reads
  it as a rate of its mixed value. Neither touches "26/7", which the extractor
  reads as twenty-six sevenths;
* in a UK corpus the cue reader never returns "falls"; in every other corpus, and
  in text with no citation path, it does;
* the title filter drops a condition word only inside a UK cross-reference's
  title. Operative text, a parenthesis that opens with a limiting word, one with a
  full stop or semicolon and space before a capital or an opening parenthesis, and
  every other corpus keep their cues;
* the cue filters only remove: the filtered cues form a subset of the unfiltered
  cues, and the difference holds only the cues they target;
* against a decimal runtime value, a Money assertion of a whole number below 2^53
  compares as its float does and as the test runner compares it; an integer
  runtime value against an integer assertion stays exact.

test_known_gap_limbs_with_no_cue_of_their_own pins a known gap. Regulation
14(3)(b) states a condition in wording no cue reads, so a model that drops it
passes this gate. test_danish_and_german_extractors_keep_a_mixed_percentages_parts
pins a known limit: in corpora under de/ and dk/ the numeric extractor reads
`2 6/7 %` as separate numbers, so the gate keeps them while it reads no division.
test_a_numeral_with_a_fraction_slash_has_no_parts pins another: the extractor
reads a numeral written with the fraction slash whole, as the number 2.857, and
before some markers, "%", "per cent" and "p.c." among them, also as the rate.
The gate keeps those readings and drops the parts, with any marker or none.
test_a_vulgar_fraction_before_a_percent_sign_keeps_its_parts pins a third: the
extractor reads `2 ½%` as two numbers, and the gate keeps both.
test_a_line_break_inside_per_cent_keeps_the_old_reading pins a fourth: the
extractor reads `2 6/7 per` and `cent` on two lines as one rate, but the gate
keeps the slash and the parts.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import math
import re
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "uk/regulation/ssi/2012/319/14"
FIXTURES = Path(__file__).parent / "fixtures" / "source_completeness"
REGULATION_14 = (FIXTURES / "uk_ssi_2012_319_14_source.txt").read_text()
PARAGRAPH_LIMBS = (FIXTURES / "uk_ssi_2012_319_14_paragraph_limbs.yaml").read_text()
_NEVER = re.compile(r"(?!)")
GROUNDING = functools.partial(
    vp.extract_typed_numeric_occurrences_from_text, profile="legacy"
)
PROPERTY_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

UK_CORPORA = (
    CITATION,
    "uk-adur/manual/council-tax-reduction-scheme-2026-2027/page-8",
)
GERMAN_CORPORA = (
    "de/statute/estg/32a",
    "de-by/statute/x/1",
    "ch/sr/642.11/36",
    "ch-zh/statute/x/1",
    "at/gesetz/estg/33",
    "at-wien/statute/x/1",
)
OTHER_ENGLISH_CORPORA = (
    "us/statute/26/32",
    "nz/act/2007/97/MA1",
    "ca/statute/income-tax-act/118",
)
# Every separator str.splitlines treats as a line boundary.
LINE_BREAKS = (
    "\n",
    "\r",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    chr(0x2028),
    chr(0x2029),
)


def occurrences(
    text: str, extractor: sc.NumericOccurrenceExtractor = GROUNDING
) -> tuple[sc.NumericOccurrenceLike, ...]:
    branch = sc.SourceStructureBranch(
        (), "formula-clause", "clause", text, 0, len(text)
    )
    return sc._formula_branch_computation_occurrences(
        branch, interval=None, extract_numeric_occurrences=extractor
    )


@contextlib.contextmanager
def mixed_percentage_collapse_disabled() -> Iterator[None]:
    saved = sc._WHOLE_MIXED_NUMERAL
    sc._WHOLE_MIXED_NUMERAL = _NEVER
    try:
        yield
    finally:
        sc._WHOLE_MIXED_NUMERAL = saved


def cues(text: str, citation: str = "") -> list[tuple[int, int, str]]:
    with sc._condition_cue_scope(citation):
        matches = sc._source_exception_or_applicability_matches(text)
    return [(match.start(), match.end(), match.group(0).lower()) for match in matches]


mixed_numbers = st.integers(min_value=2, max_value=20).flatmap(
    lambda denominator: st.tuples(
        st.integers(min_value=1, max_value=99),
        st.integers(min_value=1, max_value=denominator - 1),
        st.just(denominator),
    )
)
percent_markers = st.sampled_from([" per cent", " percent", "%", " per centum"])


@PROPERTY_SETTINGS
@given(
    number=mixed_numbers,
    marker=percent_markers,
    tail=st.sampled_from([" of the difference between A and C", " a year", ""]),
)
def test_digit_mixed_percentage_is_one_rate(
    number: tuple[int, int, int], marker: str, tail: str
) -> None:
    whole, numerator, denominator = number
    numeral = f"{whole} {numerator}/{denominator}{marker}"
    text = f"amount B is {numeral}{tail}."
    start = text.index(numeral)
    end = start + len(numeral)
    inside = [
        item for item in occurrences(text) if item.start < end and start < item.end
    ]
    assert {(item.start, item.end) for item in inside} == {(start, end)}
    expected = (whole + numerator / denominator) / 100
    assert any(math.isclose(item.value, expected) for item in inside)


@PROPERTY_SETTINGS
@given(number=mixed_numbers, marker=percent_markers)
def test_mixed_percentage_collapse_only_removes_its_own_parts(
    number: tuple[int, int, int], marker: str
) -> None:
    whole, numerator, denominator = number
    numeral = f"{whole} {numerator}/{denominator}{marker}"
    # Each neighbour yields occurrences nested in another, which the drop must
    # leave alone: "1 1/2" yields 1, 1 and 2 inside 1.5, and "10 per cent"
    # yields 0.1 over "10" inside 0.1 over "10 per cent".
    text = (
        f"pay 1 1/2 times 17 pounds, then {numeral} of the excess over 300"
        " and 5 to 10 per cent of the rest"
    )
    start = text.index(numeral)
    end = start + len(numeral)
    collapsed = occurrences(text)
    with mixed_percentage_collapse_disabled():
        everything = occurrences(text)
    assert set(collapsed) <= set(everything)
    removed = set(everything) - set(collapsed)
    # The extractor yields the parts only beside a worded marker.
    assert removed or marker == "%"
    assert all(start <= item.start and item.end <= end for item in removed)
    assert all((item.start, item.end) != (start, end) for item in removed)
    assert {item.value for item in collapsed} >= {1.0, 1.5, 2.0, 17.0, 300.0}


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("two and one-half", "2 1/2"),
        ("two and six-sevenths", "2 6/7"),
        ("twelve and one-half", "12 1/2"),
        ("one and three-quarters", "1 3/4"),
    ],
)
def test_digit_and_worded_mixed_percentages_agree(words: str, digits: str) -> None:
    worded = occurrences(f"amount B is {words} per cent of the excess")
    digit = occurrences(f"amount B is {digits} per cent of the excess")
    assert len(worded) == len(digit) == 1
    assert math.isclose(worded[0].value, digit[0].value)


@pytest.mark.parametrize(
    "text",
    [
        "amount B is 2 6 / 7 per cent of the difference",
        "amount B is 1.5 3/4 per cent of the difference",
        "amount B is 3/4 1/2 per cent of the difference",
        "amount B is 2 6/7 per centimetre of the difference",
        "amount B is 2 6/7 percentile of the difference",
        *(f"amount B is 2{line}6/7 per cent of the difference" for line in LINE_BREAKS),
        *(f"amount B is 2{line}½ per cent of the difference" for line in LINE_BREAKS),
    ],
)
def test_a_broken_or_spaced_mixed_percentage_keeps_its_old_reading(text: str) -> None:
    def reading() -> set[tuple[int, int, float]]:
        return {(item.start, item.end, item.value) for item in occurrences(text)}

    collapsed = reading()
    with mixed_percentage_collapse_disabled():
        assert collapsed == reading()
    if "/" in text:
        assert "divide" in sc._formula_operation_kinds(text)


@pytest.mark.parametrize("before", ["-", "1-"])
@pytest.mark.parametrize("marker", ["%", " per cent"])
def test_a_mixed_percentage_after_a_hyphen_is_one_rate(
    before: str, marker: str
) -> None:
    # The extractor reads some hyphens as a minus sign; either way the numeral is
    # one rate, and the gate reads no division and no parts in it.
    lead = f"amount B is {before}"
    text = f"{lead}2 6/7{marker} of the difference"
    start, end = len(lead), len(lead) + len("2 6/7")
    assert "divide" not in sc._formula_operation_kinds(text)
    inside = [
        item for item in occurrences(text) if item.start < end and start < item.end
    ]
    assert len(inside) == 1
    assert math.isclose(abs(inside[0].value), (2 + 6 / 7) / 100)


@pytest.mark.parametrize(
    ("marker", "rate"),
    [
        (" per cent", True),
        (" percent", True),
        ("%", True),
        ("-per cent", False),
        ("-percent", False),
        (" p.c.", True),
        (" per-cent", True),
        (" percentage points", False),
        ("\nper cent", False),
        (" pct", False),
        ("", False),
    ],
)
def test_a_numeral_with_a_fraction_slash_has_no_parts(marker: str, rate: bool) -> None:
    # The extractor reads a numeral with a fraction slash whole: as a plain
    # number, and before some markers also as a rate. Either way its parts go.
    numeral = f"2 6{chr(0x2044)}7"
    text = f"amount B is {numeral}{marker} of the excess"
    start = text.index(numeral)
    found = occurrences(text)
    assert {(item.start, item.end) for item in found} == {(start, start + len(numeral))}
    assert any(item.has_rate_context for item in found) is rate


@pytest.mark.parametrize(
    "numeral",
    ["2 ½ per cent", "12 ¼ percent", "2½ per cent", "2  ½ per cent"],
)
def test_a_vulgar_fraction_before_a_word_marker_is_one_rate(numeral: str) -> None:
    text = f"amount B is {numeral} of the excess"
    start = text.index(numeral)
    end = start + len(numeral)
    assert {
        (item.start, item.end)
        for item in occurrences(text)
        if item.start < end and start < item.end
    } == {(start, end)}


def test_a_vulgar_fraction_before_a_percent_sign_keeps_its_parts() -> None:
    # The extractor reads "2 ½%" as two numbers, so the gate keeps both.
    text = "amount B is 2 ½% of the excess"
    collapsed = occurrences(text)
    with mixed_percentage_collapse_disabled():
        assert collapsed == occurrences(text)
    assert [(item.start, item.end) for item in collapsed] == [(12, 13), (14, 15)]


@PROPERTY_SETTINGS
@given(number=mixed_numbers, marker=percent_markers)
def test_mixed_percentage_slash_is_no_division(
    number: tuple[int, int, int], marker: str
) -> None:
    whole, numerator, denominator = number
    numeral = f"{whole} {numerator}/{denominator}{marker}"
    assert "divide" not in sc._formula_operation_kinds(f"{numeral} of the difference")
    assert "divide" in sc._formula_operation_kinds(
        f"{numeral} of the difference divided by 7"
    )


@pytest.mark.parametrize(
    "numeral",
    [
        "2 6/7 %",
        "2 6/7  %",
        "2 6/7 per  cent",
        "2 6/7 per\tcent",
        "2 6/7  per cent",
        "2  6/7 per cent",
        "2\t6/7%",
        "2\x1f6/7 per cent",
        "2 6/7 PER CENT",
        "2 6/7 per centum",
        "2 6/7 Per Centum",
        "2 1/2 percentage points",
        "2 6/7\nper cent",
        "2 6/7\n%",
        "2 6/7-per cent",
    ],
)
def test_a_mixed_percentage_may_space_word_or_capitalize_its_marker(
    numeral: str,
) -> None:
    text = f"amount B is {numeral} of the difference"
    start = text.index(numeral)
    digits = re.match(r"\d+\s+\d+/\d+", numeral)
    assert digits is not None
    end = start + digits.end()
    assert "divide" not in sc._formula_operation_kinds(text)
    assert not sc.source_states_explicit_computation(f"The rate is {numeral}.")
    assert not [
        item for item in occurrences(text) if start <= item.start and item.end <= end
    ]


@pytest.mark.parametrize(
    "text",
    [
        "income / 2",
        "$5/person",
        "1/2 of income",
        "the 2 6/7 ratio",
        "on 15 3/7 per person",
        "amount B is 1.5 3/4 per cent of the difference",
        "amount B is 1.25 3/4 per cent of the difference",
        "amount B is 3/4 1/2 per cent of the difference",
        "amount B is 1,5 3/4 per cent of the difference",
        "amount B is 2 6 / 7 per cent of the difference",
        "amount B is 12/7 per cent of the difference",
        "amount B is 2 6/7 per centimetre of the difference",
        "amount B is 2 6/7 percentile of the difference",
        "amount B is 2 6/7 per\ncent of the difference",
        "amount B is 2 6/7percent of the difference",
        "amount B is 2 6/7\n\n% of the difference",
        "amount B is 2 6/7\x0c% of the difference",
        *(f"amount B is 2{line}6/7 per cent of the difference" for line in LINE_BREAKS),
    ],
)
def test_other_slashes_stay_divisions(text: str) -> None:
    assert "divide" in sc._formula_operation_kinds(text)


def test_a_mixed_number_with_no_rate_keeps_its_parts() -> None:
    assert {item.value for item in occurrences("the sum of 3 1/2 times the rate")} == {
        1.0,
        2.0,
        3.0,
        3.5,
    }


@PROPERTY_SETTINGS
@given(number=mixed_numbers, marker=percent_markers)
def test_a_stated_mixed_percentage_is_no_computation(
    number: tuple[int, int, int], marker: str
) -> None:
    whole, numerator, denominator = number
    numeral = f"{whole} {numerator}/{denominator}{marker}"
    assert not sc.source_states_explicit_computation(f"The rate is {numeral}.")
    assert sc.source_states_explicit_computation(f"The rate is {numeral}, plus 4 / 5.")


def test_mixed_percentage_masking_keeps_offsets() -> None:
    text = "amount B is 2 6/7 per cent of X and 4/5"
    masked = sc._without_digit_mixed_percentage_slashes(text)
    assert len(masked) == len(text)
    assert masked == "amount B is 2 6 7 per cent of X and 4/5"


@pytest.mark.parametrize(
    "text",
    [
        "3/4 1/2 per cent",
        "1.5 3/4 per cent",
        "1,5 3/4 per cent",
        "2 6 / 7 per cent",
        "12/7 per cent",
        "2 6/7 per centimetre",
        "2 6/7 per\ncent",
        "2 6/7percent",
        "2 6/7\x0c%",
        "2\x1f6/7%",
        *(f"2{line}6/7 per cent" for line in LINE_BREAKS),
    ],
)
def test_the_mask_leaves_every_other_numeral_alone(text: str) -> None:
    assert sc._without_digit_mixed_percentage_slashes(text) == text


WHOLE_FRACTION_GAPS = (
    " ",
    "  ",
    "\t",
    "\xa0",
    "\x1f",
    chr(0x1680),
    chr(0x2000),
    chr(0x2003),
    chr(0x200A),
    chr(0x202F),
    chr(0x205F),
    chr(0x3000),
    *LINE_BREAKS,
)
MARKER_GAPS = ("", " ", "  ", "\t", "\xa0", "\r\n", "\n\n", "-", *LINE_BREAKS)
MARKERS = (
    "%",
    "percent",
    "per cent",
    "per  cent",
    "per\tcent",
    "per\ncent",
    "per\xa0cent",
    "per-cent",
    "per centum",
    "Per Centum",
    "percentage points",
    "per centimetre",
    "percentile",
    "PER CENT",
    "p.c.",
    "pct",
)
BEFORE_NUMERAL = (
    "",
    "£",
    "(",
    "-",
    "1-",
    chr(0x2212),
    f"1{chr(0x2013)}",
    "1.",
    "3/",
    "1,",
    "x ",
)


@PROPERTY_SETTINGS
@given(
    number=mixed_numbers,
    before=st.sampled_from(BEFORE_NUMERAL),
    gap=st.sampled_from(WHOLE_FRACTION_GAPS),
    marker_gap=st.sampled_from(MARKER_GAPS),
    marker=st.sampled_from(MARKERS),
)
def test_the_mask_and_the_collapse_agree(
    number: tuple[int, int, int],
    before: str,
    gap: str,
    marker_gap: str,
    marker: str,
) -> None:
    # The slash mask reads text, and the collapse reads the occurrences of the
    # extractor the gate passes it. Where the mask reads no division, no part of
    # the numeral stays; where it reads one, the collapse drops nothing.
    whole, numerator, denominator = number
    digits = f"{whole}{gap}{numerator}/{denominator}"
    lead = f"amount B is {before}"
    text = f"{lead}{digits}{marker_gap}{marker} of the difference"
    start, end = len(lead), len(lead) + len(digits)
    collapsed = occurrences(text)
    with mixed_percentage_collapse_disabled():
        everything = occurrences(text)
    if sc._without_digit_mixed_percentage_slashes(text) != text:
        assert not [
            item for item in collapsed if start <= item.start and item.end <= end
        ]
    else:
        assert collapsed == everything


# Every character Python's \s matches that str.splitlines does not break on.
SAME_LINE_SPACES = tuple(
    char
    for char in map(chr, range(0x110000))
    if re.fullmatch(r"\s", char) and char not in LINE_BREAKS
)
# The extractor reads "per centimetre" and "percentile" as "per cent" and
# "percent"; the gate takes neither as a marker. The extractor also reads across
# a line break inside "per cent", where the gate keeps its old reading.
DIFFERENTIAL_MARKERS = tuple(
    marker
    for marker in MARKERS
    if marker not in ("per centimetre", "percentile", "per\ncent")
)


def mixed_number_texts(
    fraction: str, gaps: tuple[str, ...]
) -> Iterator[tuple[str, int, int]]:
    # Each whole-fraction gap before each marker, then each gap before the
    # marker after a plain space between whole number and fraction.
    lead = "amount B is "
    marker_gaps = ("", "  ", "-", "\r\n", "\n\n", *SAME_LINE_SPACES, *LINE_BREAKS)
    for gap, marker_gap, marker in itertools.chain(
        itertools.product(gaps, [" "], DIFFERENTIAL_MARKERS),
        itertools.product([" "], marker_gaps, DIFFERENTIAL_MARKERS),
    ):
        numeral = f"2{gap}{fraction}"
        text = f"{lead}{numeral}{marker_gap}{marker} of the difference"
        yield text, len(lead), len(lead) + len(numeral)


def extractor_reads_whole(
    text: str, start: int, end: int, *, rate: bool, value: float
) -> bool:
    # One occurrence covers the numeral and holds its value as a mixed number,
    # or that value as a rate: with no gap, "26/7" is twenty-six sevenths, not
    # two and six sevenths. The value marks a rate, since the extractor flags no
    # rate context when "per" and "cent" have a space other than " " or tab
    # between them.
    values = (value / 100,) if rate else (value, value / 100)
    return any(
        item.start <= start
        and end <= item.end
        and any(math.isclose(item.value, expected) for expected in values)
        for item in GROUNDING(text)
    )


def collapse_matches(text: str, start: int, end: int) -> bool:
    return any(
        match.span() == (start, end) for match in sc._WHOLE_MIXED_NUMERAL.finditer(text)
    )


def test_on_one_line_the_mask_hides_a_slash_where_the_extractor_reads_a_rate() -> None:
    gaps = (*SAME_LINE_SPACES, "", "  ", " \t", chr(0x200B), "_")
    disagreements = [
        text
        for text, start, end in mixed_number_texts("6/7", gaps)
        if (sc._without_digit_mixed_percentage_slashes(text) != text)
        != extractor_reads_whole(text, start, end, rate=True, value=2 + 6 / 7)
    ]
    assert disagreements == []


@pytest.mark.parametrize(
    "fraction",
    [
        f"6{chr(0x2044)}7",
        f"6 {chr(0x2044)} 7",
        f"6\t{chr(0x2044)}\t7",
        f"6{chr(0x2044)} 7",
        f"6 {chr(0x2044)}7",
        f"6\n{chr(0x2044)}\n7",
    ],
)
def test_the_collapse_reads_a_fraction_slash_numeral_whole(fraction: str) -> None:
    # The extractor reads every mixed number with the fraction slash whole,
    # with any whitespace around the slash, so the collapse matches one before
    # any marker.
    gaps = (*SAME_LINE_SPACES, "", "  ", chr(0x200B), "_")
    disagreements = [
        text
        for text, start, end in mixed_number_texts(fraction, gaps)
        if collapse_matches(text, start, end)
        != extractor_reads_whole(text, start, end, rate=False, value=2 + 6 / 7)
    ]
    assert disagreements == []


def test_the_collapse_reads_a_vulgar_rate_whole_as_the_extractor_does() -> None:
    gaps = (*SAME_LINE_SPACES, "", "  ", chr(0x200B), "_")
    disagreements = [
        text
        for text, start, end in mixed_number_texts(chr(0xBD), gaps)
        if collapse_matches(text, start, end)
        != extractor_reads_whole(text, start, end, rate=True, value=2.5)
    ]
    assert disagreements == []


@pytest.mark.parametrize("space", SAME_LINE_SPACES)
@pytest.mark.parametrize("numeral", ["2 6/7", f"2 {chr(0xBD)}"])
def test_any_space_on_one_line_may_split_per_cent(space: str, numeral: str) -> None:
    text = f"amount B is {numeral} per{space}cent of the excess"
    start = text.index(numeral)
    end = start + len(numeral)
    assert "divide" not in sc._formula_operation_kinds(text)
    assert [(item.start, item.end) for item in occurrences(text)] == [
        (start, end + len(f" per{space}cent"))
    ]


@pytest.mark.parametrize("line_break", LINE_BREAKS)
def test_a_line_break_inside_per_cent_keeps_the_old_reading(line_break: str) -> None:
    text = f"amount B is 2 6/7 per{line_break}cent of the excess"
    assert "divide" in sc._formula_operation_kinds(text)
    assert {7.0, 6.0, 2.0} <= {item.value for item in occurrences(text)}


@pytest.mark.parametrize(
    "text",
    [
        "under regulation 14, amount B is 2 6/7 per cent of the excess",
        "amount B (see regulation 14) is 2 6/7 per cent of the excess",
    ],
)
def test_the_drop_finds_the_numeral_in_the_text_the_extractor_reads(
    text: str,
) -> None:
    # The recall text drops the citation, so its offsets differ from the
    # branch's. The drop must find the numeral where the extractor read it.
    assert len(sc.authoritative_numeric_recall_text(text)) < len(text)
    found = occurrences(text)
    assert len(found) == 1
    assert math.isclose(found[0].value, (2 + 6 / 7) / 100)


@pytest.mark.parametrize("profile", ["de-DE", "da-DK"])
def test_danish_and_german_extractors_keep_a_mixed_percentages_parts(
    profile: str,
) -> None:
    text = "amount B is 2 6/7 % of the difference"
    extractor = functools.partial(
        vp.extract_typed_numeric_occurrences_from_text, profile=profile
    )
    assert "divide" not in sc._formula_operation_kinds(text)
    assert 7.0 in {item.value for item in occurrences(text, extractor)}


@pytest.mark.parametrize(
    ("citation", "profile"),
    [
        ("de/statute/estg/32a", "de-DE"),
        ("dk/statute/x/1", "da-DK"),
        ("de-by/statute/x/1", "legacy"),
        ("at/gesetz/estg/33", "legacy"),
        ("ch/sr/642.11/36", "legacy"),
        (CITATION, "legacy"),
    ],
)
def test_only_corpora_under_de_and_dk_read_numbers_in_their_own_locale(
    citation: str, profile: str
) -> None:
    assert vp._numeric_profile_for_citation_path(citation) == profile


@PROPERTY_SETTINGS
@given(number=mixed_numbers, marker=percent_markers)
def test_every_arithmetic_reader_reads_a_mixed_percentage_as_one_rate(
    number: tuple[int, int, int], marker: str
) -> None:
    whole, numerator, denominator = number
    stated = f"The rate is {whole} {numerator}/{denominator}{marker}"
    assert not sc._source_states_nonrounding_computation(f"{stated}.")
    assert not sc._formula_clause_states_substantive_operation(f"{stated}.")
    assert (
        sc._rounding_only_direction(f"{stated}, rounded down to the nearest penny.")
        == "downward"
    )
    assert sc._source_states_nonrounding_computation(f"{stated}, plus 4 / 5.")
    assert sc._formula_clause_states_substantive_operation(f"{stated}, plus 4 / 5.")


FALLS_CLAUSES = (
    "the day falls within a week in respect of which the person has no income",
    "the week in which the day falls and the applicable amount",
    "a resident who falls to be disregarded for the purposes of discount",
    "income that falls short of the amount",
    "Falls Church is the place of residence",
)


@pytest.mark.parametrize("citation", UK_CORPORA)
@pytest.mark.parametrize("clause", FALLS_CLAUSES)
def test_the_english_verb_is_no_condition_in_a_uk_corpus(
    citation: str, clause: str
) -> None:
    assert [cue for cue in cues(clause, citation) if cue[2] == "falls"] == []


@pytest.mark.parametrize("citation", OTHER_ENGLISH_CORPORA)
def test_other_english_corpora_keep_the_cue(citation: str) -> None:
    assert [cue[2] for cue in cues(FALLS_CLAUSES[0], citation)] == ["falls"]


@pytest.mark.parametrize(
    "citation",
    ("ukraine/law/1", "UK/regulation/ssi/2012/319/14", "uk_old/x/1", "gb/x/1"),
)
def test_a_namespace_that_only_resembles_uk_keeps_the_cue(citation: str) -> None:
    assert [cue[2] for cue in cues(FALLS_CLAUSES[0], citation)] == ["falls"]


@pytest.mark.parametrize("citation", GERMAN_CORPORA)
def test_german_falls_stays_a_condition(citation: str) -> None:
    text = "Der Betrag erhöht sich, falls der Antrag rechtzeitig gestellt wird."
    assert [cue[2] for cue in cues(text, citation)] == ["falls"]


def test_falls_stays_a_cue_when_no_citation_names_a_language() -> None:
    assert [cue[2] for cue in cues(FALLS_CLAUSES[0])] == ["falls"]


@PROPERTY_SETTINGS
@given(
    citation=st.sampled_from(UK_CORPORA),
    before=st.sampled_from(["if the day", "the day", "unless the week"]),
    after=st.sampled_from(["within a week", "to be disregarded", "if income is low"]),
)
def test_falls_filter_only_removes_falls(
    citation: str, before: str, after: str
) -> None:
    text = f"{before} falls {after}"
    scoped = cues(text, citation)
    unscoped = cues(text)
    assert set(scoped) <= set(unscoped)
    assert {cue[2] for cue in set(unscoped) - set(scoped)} == {"falls"}


CUES = (
    "subject to",
    "does not apply",
    "if",
    "except",
    "unless",
    "notwithstanding",
)
titles = st.builds(
    lambda head, cue, tail: f"{head} {cue} {tail}",
    st.sampled_from(
        [
            "persons not entitled to council tax reduction: persons",
            "residents of a dwelling to whom regulation 14(6)",
            "circumstances where the condition",
        ]
    ),
    st.sampled_from(CUES),
    st.sampled_from(["immigration control", "the applicant", "work-focused interview"]),
)
reference_forms = st.sampled_from(
    [
        "regulation 57 ({0})",
        "regulation 14A ({0})",
        "regulation 14(6) ({0})",
        "Schedule 2 ({0})",
        "paragraph 12 ({0})",
        "section 75 ({0})",
    ]
)


def title_cues(text: str, title: str, citation: str) -> list[tuple[int, int, str]]:
    start = text.index(title)
    end = start + len(title)
    return [cue for cue in cues(text, citation) if start <= cue[0] and cue[1] <= end]


@PROPERTY_SETTINGS
@given(
    title=titles,
    form=reference_forms,
    citation=st.sampled_from(UK_CORPORA),
)
def test_condition_word_in_a_uk_reference_title_is_no_cue(
    title: str, form: str, citation: str
) -> None:
    reference = form.format(title)
    text = f"a person described in {reference} is entitled if the claim stands"
    assert title_cues(text, title, citation) == []
    assert ("if" in [cue[2] for cue in cues(text, citation)]) is True
    remaining = cues(text, citation)
    assert text[remaining[-1][0] : remaining[-1][1]] == "if"
    assert remaining[-1][0] == text.rindex(" if the claim") + 1


@PROPERTY_SETTINGS
@given(first=titles, second=titles, third=titles)
def test_condition_words_in_a_titled_list_are_no_cues(
    first: str, second: str, third: str
) -> None:
    text = (
        f"is not a person to whom regulation 15 ({first}), regulation 16 ({second}) "
        f"or 19 ({third}) applies"
    )
    assert cues(text, CITATION) == []


@PROPERTY_SETTINGS
@given(
    title=titles,
    form=reference_forms,
    citation=st.sampled_from(["us/statute/26/32", "nz/act/2007/97/MA1"]),
)
def test_other_corpora_keep_cues_inside_titles(
    title: str, form: str, citation: str
) -> None:
    text = f"a person described in {form.format(title)} is entitled"
    assert title_cues(text, title, citation) == title_cues(text, title, "")
    assert title_cues(text, title, citation)


@pytest.mark.parametrize(
    ("reference", "cue"),
    [
        ("paragraph 7 (except paragraph 112)", "except"),
        ("Schedule 2A (except paragraphs 2(1)(b) and 3(1)(b))", "except"),
        ("paragraph 26 (subject to paragraph 26(1)(b)(i))", "subject to"),
        ("regulation 5 (or, if paragraph (3) applies, regulation 6)", "if"),
        ("regulation 5 (other than paragraph (b) if it applies)", "if"),
        ("section 7 (unless paragraph 3 applies)", "unless"),
        ("section 7 (if any)", "if"),
        ("section 7 (notwithstanding paragraph 3)", "notwithstanding"),
    ],
)
def test_limiting_parenthesis_keeps_its_cue(reference: str, cue: str) -> None:
    assert cue in [item[2] for item in cues(f"under {reference}", CITATION)]


@pytest.mark.parametrize(
    "opening",
    [
        "excluding paragraph 3",
        "save paragraph 3",
        "apart from paragraph 3",
        "but paragraph 3",
        "and paragraph 3",
        "so far as paragraph 3 applies",
        "in so far as paragraph 3 applies",
        "insofar as paragraph 3 applies",
        "to the extent paragraph 3 applies",
        "where paragraph 3 applies",
        "with the exception of paragraph 3",
        "only if paragraph 3 applies",
        "provided that paragraph 3 applies",
        "read with paragraph 3",
        "as modified by paragraph 3",
    ],
)
def test_a_parenthesis_that_opens_by_limiting_the_reference_keeps_its_cues(
    opening: str,
) -> None:
    text = f"under section 7 ({opening} if it applies)"
    assert cues(text, CITATION) == cues(text)
    assert "if" in [item[2] for item in cues(text, CITATION)]


@pytest.mark.parametrize(
    "text",
    [
        "See paragraph 5 (average income. The amount is nil if absent.) End.",
        "See paragraph 5 (average income; The amount is nil if absent.) End.",
        "See paragraph 5 (average income. (1) The amount is nil if absent) End.",
        "See paragraph 5 (" + "q" * 301 + " if absent) End.",
    ],
)
def test_a_parenthesis_left_open_keeps_its_cues(text: str) -> None:
    assert cues(text, CITATION) == cues(text)
    assert "if" in [item[2] for item in cues(text, CITATION)]


@pytest.mark.parametrize("line_break", LINE_BREAKS)
def test_a_parenthesis_after_a_line_break_keeps_its_cues(line_break: str) -> None:
    text = (
        f"See regulation 57{line_break}"
        "(residents to whom regulation 14(6) does not apply)."
    )
    assert "does not apply" in [item[2] for item in cues(text, CITATION)]


def test_a_cue_that_runs_past_the_title_stays() -> None:
    text = "See regulation 57 (gilt for residents) nicht."
    assert [item[2] for item in cues(text, CITATION)] == ["gilt for residents) nicht"]


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        (
            "regulation 14(6)(a) (persons to whom regulation 5 does not apply)",
            "does not apply",
        ),
        ("regulation 14(6A) (persons if absent)", "if"),
        ("regulation 2(lxxxviii) (persons if absent)", "if"),
        ("regulation 7(circumstances if absent)", "if"),
    ],
)
def test_a_title_may_follow_subdivisions_or_touch_its_number(
    text: str, cue: str
) -> None:
    assert cue in [item[2] for item in cues(text)]
    assert cue not in [item[2] for item in cues(text, CITATION)]


@pytest.mark.parametrize(
    "text",
    [
        "regulation 7(circumstances) (persons if absent)",
        "regulation 2(clxxxviii) (persons if absent)",
        "regulation 7(2 or 3) (persons if absent)",
    ],
)
def test_a_long_or_spaced_parenthesis_on_a_number_is_no_subdivision(
    text: str,
) -> None:
    assert "if" in [item[2] for item in cues(text, CITATION)]


@pytest.mark.parametrize(
    "text",
    [
        "under subregulation 14 (persons if absent) the amount applies",
        "regulation14 (persons if absent)",
    ],
)
def test_a_keyword_inside_a_word_or_touching_its_number_opens_no_title(
    text: str,
) -> None:
    assert "if" in [item[2] for item in cues(text, CITATION)]


@pytest.mark.parametrize(
    "text",
    [
        "under paragraph (2) (persons if absent) the amount applies",
        "paragraph (3) (amount of reduction if a couple) applies",
    ],
)
def test_a_reference_with_its_number_in_parentheses_opens_no_title(
    text: str,
) -> None:
    assert "if" in [item[2] for item in cues(text, CITATION)]


def test_a_reference_number_may_carry_several_letters() -> None:
    text = "under section 13ZA (persons if absent) the amount applies"
    assert "if" not in [item[2] for item in cues(text, CITATION)]
    assert "if" in [item[2] for item in cues(text)]


def test_the_title_filter_stops_at_the_end_of_a_reference_list() -> None:
    text = (
        "See regulation 15 (absentees) or 19 (students). "
        "A sum of 5 (persons if absent) applies."
    )
    assert "if" in [item[2] for item in cues(text, CITATION)]


def test_title_that_opens_with_when_is_a_title() -> None:
    text = "regulation 58 (when entitlement begins and change of circumstances)"
    assert cues(text, CITATION) == []
    assert cues(text) != []


@PROPERTY_SETTINGS
@given(title=titles, form=reference_forms)
def test_title_filter_only_removes_cues_inside_titles(title: str, form: str) -> None:
    text = f"if a person in {form.format(title)} applies, unless excluded"
    scoped = cues(text, CITATION)
    unscoped = cues(text)
    assert set(scoped) <= set(unscoped)
    start = text.index(title)
    end = start + len(title)
    assert all(start <= cue[0] and cue[1] <= end for cue in set(unscoped) - set(scoped))
    assert {cue[2] for cue in scoped} == {"if", "unless"}


def test_condition_cue_scope_restores_and_nests() -> None:
    assert sc._CONDITION_CUE_CITATION_PATH.get() == ""
    with sc._condition_cue_scope(CITATION):
        with sc._condition_cue_scope(""):
            assert sc._CONDITION_CUE_CITATION_PATH.get() == CITATION
        with sc._condition_cue_scope("de/statute/estg/32a"):
            assert sc._CONDITION_CUE_CITATION_PATH.get() == "de/statute/estg/32a"
        assert sc._CONDITION_CUE_CITATION_PATH.get() == CITATION
    assert sc._CONDITION_CUE_CITATION_PATH.get() == ""


def test_the_analysis_reads_every_cue_in_the_scope_of_its_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scopes: list[str] = []
    matcher = sc._source_exception_or_applicability_matches

    def recording_matcher(*args: object, **kwargs: object) -> object:
        scopes.append(sc._CONDITION_CUE_CITATION_PATH.get())
        return matcher(*args, **kwargs)

    monkeypatch.setattr(
        sc, "_source_exception_or_applicability_matches", recording_matcher
    )
    analyze(SIX_CASES)

    assert scopes
    assert set(scopes) == {CITATION}


def test_exception_branches_read_the_citation_they_are_given() -> None:
    text = "A resident who falls to be disregarded is entitled."
    branches = sc.recognize_source_structure(text, corpus_citation_path=CITATION)
    assert (
        sc._source_exception_branches(
            text,
            branches=branches,
            active_branches=branches,
            deferred_paths=set(),
            corpus_citation_path=CITATION,
        )
        == ()
    )
    assert sc._source_exception_branches(
        text,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        corpus_citation_path="de/statute/estg/32a",
    )


BASE_INPUT = {
    "relevant_person_is_liable_to_pay_council_tax_under_section_75": True,
    "relevant_person_is_liable_for_council_tax_for_dwelling_as_sole_or_main_residence": True,
    "regulation_15_applies": False,
    "regulation_16_applies": False,
    "regulation_19_applies": False,
    "relevant_person_has_made_application_in_accordance_with_part_9": True,
    "appropriate_maximum_council_tax_reduction_exists": True,
    "relevant_person_has_no_income": True,
    "relevant_person_weekly_income": 0,
    "applicable_amount": 300,
    "appropriate_maximum_council_tax_reduction": 100,
    "no_included_other_resident_is_liable_to_pay_rent_to_relevant_person": True,
    "relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income": True,
    "appropriate_alternative_maximum_council_tax_reduction": 120,
    "relevant_person_entitlement_to_council_tax_reduction_has_commenced": True,
}
ROUTE_14A_ONLY = (
    "relevant_person_is_liable_for_council_tax_for_dwelling",
    "dwelling_is_in_band_e_f_g_or_h_and_is_sole_or_main_residence",
    "relevant_person_is_not_member_of_couple_or_polygamous_marriage",
    "relevant_person_is_not_responsible_for_child_or_young_person",
)
MODULE = "uk:regulations/ssi/2012/319/14"
RATE = 0.02857142857142857142857142857
EXCESS_INCOME = {
    "relevant_person_has_no_income": False,
    "relevant_person_weekly_income": 335,
    "relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income": False,
    "appropriate_alternative_maximum_council_tax_reduction": 0,
}


def judgment(value: bool) -> str:
    return "holds" if value else "not_holds"


def model_outputs(facts: dict[str, object]) -> dict[str, object]:
    """What the paragraph-limb model computes, stated independently of the gate."""

    common = (
        facts["relevant_person_is_liable_to_pay_council_tax_under_section_75"]
        and facts[
            "relevant_person_is_liable_for_council_tax_for_dwelling_as_sole_or_main_residence"
        ]
        and not facts["regulation_15_applies"]
        and not facts["regulation_16_applies"]
        and not facts["regulation_19_applies"]
        and facts["relevant_person_has_made_application_in_accordance_with_part_9"]
    )
    income = facts["relevant_person_weekly_income"]
    applicable = facts["applicable_amount"]
    maximum = facts["appropriate_maximum_council_tax_reduction"]
    amount_b = RATE * max(0, income - applicable)
    paragraph_5a = facts["relevant_person_has_no_income"] or income <= applicable
    paragraph_5b = not paragraph_5a and maximum > amount_b
    income_route = facts["appropriate_maximum_council_tax_reduction_exists"] and (
        paragraph_5a or paragraph_5b
    )
    paragraph_6 = (
        facts["no_included_other_resident_is_liable_to_pay_rent_to_relevant_person"]
        and facts[
            "relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income"
        ]
    )
    limb_a = maximum if paragraph_5a else 0
    limb_b = maximum - amount_b if paragraph_5b else 0
    limb_c = (
        facts["appropriate_alternative_maximum_council_tax_reduction"]
        if paragraph_6
        else 0
    )
    before = max((limb_a + limb_b) if income_route else 0, limb_c)
    # Regulation 14A's route stays off in every case, so its amount is zero.
    paragraph_3a = before >= 0
    entitled = (
        facts["relevant_person_entitlement_to_council_tax_reduction_has_commenced"]
        and common
        and paragraph_3a
        and (income_route or paragraph_6)
    )
    return {
        "regulation_14_common_conditions_satisfied": judgment(common),
        "regulation_14_income_excess_reduction_amount": amount_b,
        "regulation_14_paragraph_5_a_applies": judgment(paragraph_5a),
        "regulation_14_paragraph_5_b_applies": judgment(paragraph_5b),
        "regulation_14_income_route_satisfied": judgment(income_route),
        "regulation_14_paragraph_6_applies": judgment(paragraph_6),
        "regulation_14_paragraph_8_a_amount": limb_a,
        "regulation_14_paragraph_8_b_amount": limb_b,
        "regulation_14_paragraph_8_c_amount": limb_c,
        "regulation_14_reduction_before_regulation_14a_comparison": before,
        "regulation_14_paragraph_3a_condition_satisfied": judgment(paragraph_3a),
        "regulation_14_reduction_entitlement": judgment(entitled),
        "regulation_14_council_tax_reduction": before if entitled else 0,
    }


def case(name: str, **overrides: object) -> dict[str, object]:
    facts = {**BASE_INPUT, **overrides}
    inputs: dict[str, object] = {
        f"{MODULE}#input.{key}": value for key, value in facts.items()
    }
    inputs.update({f"{MODULE}A#input.{key}": False for key in ROUTE_14A_ONLY})
    return {
        "name": name,
        "period": {
            "period_kind": "custom",
            "name": "day",
            "start": "2026-01-15",
            "end": "2026-01-15",
        },
        "input": inputs,
        "output": {
            f"{MODULE}#{key}": value for key, value in model_outputs(facts).items()
        },
    }


SIX_CASES = (
    case("both_routes"),
    case("regulation_15_displaces", regulation_15_applies=True),
    case("regulation_16_displaces", regulation_16_applies=True),
    case("regulation_19_displaces", regulation_19_applies=True),
    case("excess_income", **EXCESS_INCOME),
    case(
        "income_at_applicable",
        **{**EXCESS_INCOME, "relevant_person_weekly_income": 300},
    ),
)
LIMB_B_PAIR = case(
    "excess_income_small_maximum",
    **{**EXCESS_INCOME, "appropriate_maximum_council_tax_reduction": 0.5},
)
LIMB_C_PAIR = case(
    "both_routes_without_alternative",
    relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income=False,
)
# What the gate quoted before the three fixes, each anchored on the slash of the
# rate, the verb "falls" or a word inside another provision's title.
MISREAD_ANCHORS = (
    "formula clause 12",
    "subject to immigration control",
    "the day falls within",
    "falls to be disregarded",
    "regulation 14(6) does not apply",
    "the day falls and the applicable",
)


def analyze(
    cases: tuple[dict[str, object], ...], module: str = PARAGRAPH_LIMBS
) -> list[str]:
    bindings = vp.collect_artifact_numeric_bindings(
        module,
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    profile = vp._numeric_profile_for_citation_path(CITATION)
    result = sc.analyze_complete_source_unit(
        module,
        REGULATION_14,
        corpus_citation_path=CITATION,
        test_cases=list(cases),
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            vp.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return list(result.issues)


def test_fixture_module_parses() -> None:
    payload = yaml.safe_load(PARAGRAPH_LIMBS)
    assert payload["format"] == "rulespec/v1"
    assert payload["module"]["source_verification"]["corpus_citation_path"] == CITATION


def test_a_model_of_each_limb_clears_the_gate() -> None:
    assert analyze((*SIX_CASES, LIMB_B_PAIR, LIMB_C_PAIR)) == []


AMOUNT_B = f"{MODULE}#regulation_14_income_excess_reduction_amount"
MONEY_RULE = {"dtype": "Money"}
whole_amounts = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.integers(min_value=-(2**53) + 1, max_value=2**53 - 1),
)
residues = st.decimals(
    min_value=Decimal("-1e-13"),
    max_value=Decimal("1e-13"),
    places=20,
    allow_nan=False,
    allow_infinity=False,
)


def state_amount_b_as_an_integer(built: dict[str, object]) -> dict[str, object]:
    output = built["output"]
    assert isinstance(output, dict)
    if output[AMOUNT_B] != 1:
        return built
    return {**built, "output": {**output, AMOUNT_B: 1}}


def test_the_gate_takes_a_whole_amount_b_as_the_test_runner_does() -> None:
    cases = tuple(
        state_amount_b_as_an_integer(built)
        for built in (*SIX_CASES, LIMB_B_PAIR, LIMB_C_PAIR)
    )
    assert [
        built["name"]
        for built in cases
        if isinstance(built["output"], dict) and type(built["output"][AMOUNT_B]) is int
    ] == ["excess_income", "excess_income_small_maximum"]
    assert analyze(cases) == []


@PROPERTY_SETTINGS
@given(whole_amounts, residues)
def test_a_whole_money_assertion_compares_as_its_float_does(
    whole: int, residue: Decimal
) -> None:
    runtime = Decimal(whole) + residue
    assert sc._asserted_formula_runtime_values_equal(
        MONEY_RULE, runtime, whole
    ) == sc._asserted_formula_runtime_values_equal(MONEY_RULE, runtime, float(whole))


@PROPERTY_SETTINGS
@given(whole_amounts, residues)
def test_the_gate_and_the_test_runner_agree_on_a_whole_money_assertion(
    whole: int, residue: Decimal
) -> None:
    runtime = Decimal(whole) + residue
    assert sc._asserted_formula_runtime_values_equal(
        MONEY_RULE, runtime, whole
    ) == cli._rulespec_scalar_matches({"kind": "decimal", "value": str(runtime)}, whole)


@PROPERTY_SETTINGS
@given(whole_amounts, whole_amounts)
def test_an_integer_money_value_against_an_integer_assertion_stays_exact(
    runtime: int, asserted: int
) -> None:
    assert sc._asserted_formula_runtime_values_equal(MONEY_RULE, runtime, asserted) == (
        runtime == asserted
    )


@PROPERTY_SETTINGS
@given(st.integers(min_value=2**52, max_value=2**53 - 2))
def test_adjacent_integer_money_values_stay_unequal_at_one_binary64_step(
    runtime: int,
) -> None:
    assert not sc._asserted_formula_runtime_values_equal(
        MONEY_RULE, runtime, runtime + 1
    )


@pytest.mark.parametrize(
    ("cases", "demanded"),
    [
        (SIX_CASES + (LIMB_B_PAIR,), ["if paragraph (6) applies"]),
        (
            SIX_CASES,
            ["if paragraph (6) applies", "appropriate maximum council tax reduction"],
        ),
    ],
)
def test_the_gate_still_demands_the_conditions_its_cues_read(
    cases: tuple[dict[str, object], ...], demanded: list[str]
) -> None:
    issues = analyze(cases)
    assert len(issues) == 1
    assert "require paired positive/blocking cases" in issues[0]
    missing = issues[0].split("missing:", 1)[1].split("The evaluator", 1)[0]
    for condition in demanded:
        assert condition in missing
    for anchor in MISREAD_ANCHORS:
        assert anchor not in issues[0]


def test_known_gap_limbs_with_no_cue_of_their_own() -> None:
    # No cue reads "is not a person to whom", and regulation 14(3)(b) carried a
    # demand only through "subject to" in the title of regulation 19, so dropping
    # (3)(b) passes the gate. Issue 2153 tracks a cue for that wording.
    without_paragraph_3_b = PARAGRAPH_LIMBS
    for limb in (
        "and not regulation_15_applies",
        "and not regulation_16_applies",
        "and not regulation_19_applies",
    ):
        assert limb in without_paragraph_3_b
        without_paragraph_3_b = without_paragraph_3_b.replace(limb, "")
    cases_without_paragraph_3_b = tuple(
        {
            **test_case,
            "output": {
                **test_case["output"],
                f"{MODULE}#regulation_14_common_conditions_satisfied": "holds",
            },
        }
        for test_case in (*SIX_CASES, LIMB_B_PAIR, LIMB_C_PAIR)
    )

    assert analyze(cases_without_paragraph_3_b, without_paragraph_3_b) == []
