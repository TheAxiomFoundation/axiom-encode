"""The terminal session-law history strip does linear work (#1828).

The strip used to re-normalize and re-split the whole remaining suffix for each
candidate history start, so its work was text length times candidates. These
tests count the characters read through every pattern and helper the strip and
its resume helper use (`METERED_PATTERNS` and `_whitespace_run_start`), not
elapsed time, and keep main's implementation as a reference: the
fixed strip must return exactly what main returned on every input.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

FIXTURES = Path(__file__).parent / "fixtures"


class _CountingPattern:
    """A compiled pattern that adds the characters each call reads to ``work``.

    * ``sub``: the whole string.
    * ``match`` and ``fullmatch``: the longest prefix the pattern consumes from
      ``pos``, plus one character. An anchored match reads that prefix and fails
      at the next character; billing up to ``endpos`` instead would charge the
      rest of the paragraph to every candidate that reads only its first entry.
    * ``search``: from ``pos`` to the end of the match, or to ``endpos``.
    * ``finditer``: lazily, from ``pos`` to the end of each match the caller
      takes, and to ``endpos`` when the caller exhausts the iterator. A caller
      that rescans from the paragraph start is billed for the rescan.
    """

    def __init__(self, pattern: re.Pattern[str], meter: dict[str, int]):
        self._pattern = pattern
        self._meter = meter
        self.pattern = pattern.pattern
        self.flags = pattern.flags

    @staticmethod
    def _end(string: str, endpos: int | None) -> int:
        return len(string) if endpos is None else min(endpos, len(string))

    @staticmethod
    def _args(string, pos, endpos):
        return (string, pos) if endpos is None else (string, pos, endpos)

    def sub(self, repl, string, count=0):
        self._meter["work"] += len(string)
        return self._pattern.sub(repl, string, count)

    def fullmatch(self, string, pos=0, endpos=None):
        prefix = self._pattern.match(*self._args(string, pos, endpos))
        self._meter["work"] += (prefix.end() - pos if prefix else 0) + 1
        return self._pattern.fullmatch(*self._args(string, pos, endpos))

    def match(self, string, pos=0, endpos=None):
        found = self._pattern.match(*self._args(string, pos, endpos))
        self._meter["work"] += (found.end() - pos if found else 0) + 1
        return found

    def search(self, string, pos=0, endpos=None):
        found = self._pattern.search(*self._args(string, pos, endpos))
        end = found.end() if found else self._end(string, endpos)
        self._meter["work"] += max(1, end - pos)
        return found

    def finditer(self, string, pos=0, endpos=None):
        reached = pos
        for found in self._pattern.finditer(*self._args(string, pos, endpos)):
            self._meter["work"] += max(1, found.end() - reached)
            reached = found.end()
            yield found
        self._meter["work"] += max(0, self._end(string, endpos) - reached)


# Every pattern the strip and its resume helper read the text through.
METERED_PATTERNS = (
    "_SESSION_LAW_SENTENCE_SEPARATOR",
    "_SESSION_LAW_PARAGRAPH_BREAK",
    "_SESSION_LAW_TAIL_START",
    "_SESSION_LAW_ENTRY",
    "_SESSION_LAW_OPERATIVE_CONTINUATION",
    "_SESSION_LAW_OPERATIVE_CONTINUATION_AT",
    "_SESSION_LAW_CONTINUATION_WRAPPER",
)


_PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n")


# --- main's implementation, kept verbatim as the differential reference ----


def _legacy_validated_entries(history: str):
    normalized = sc._SESSION_LAW_SENTENCE_SEPARATOR.sub("; ", history)
    entries = []
    for part in normalized.split(";"):
        entry = sc._SESSION_LAW_ENTRY.fullmatch(part)
        if entry is None:
            return None
        for qualifier in ("effective", "operative"):
            month = entry.group(f"{qualifier}_month")
            if month is None:
                continue
            month_number = tuple(calendar.month_abbr).index(month[:3].title())
            try:
                date(
                    int(entry.group(f"{qualifier}_year")),
                    month_number,
                    int(entry.group(f"{qualifier}_day")),
                )
            except ValueError:
                return None
        entries.append(entry)
    return tuple(entries)


def _legacy_resume_offset(source_text: str, candidate):
    if (candidate.group("history_label") or "").lower() == "history":
        return None
    suffix = source_text[candidate.start() :]
    paragraph_break = sc._SESSION_LAW_PARAGRAPH_BREAK.search(suffix)
    paragraph = suffix[: paragraph_break.start()] if paragraph_break else suffix
    entry = sc._SESSION_LAW_ENTRY.match(paragraph)
    if entry is None or _legacy_validated_entries(entry.group()) is None:
        return None
    continuation = paragraph[entry.end() :]
    continuation = re.sub(r"^(?:[('\"“‘][ \t]*){0,2}", "", continuation)
    if sc._SESSION_LAW_OPERATIVE_CONTINUATION.match(continuation) is None:
        return None
    for paragraph_candidate in sc._SESSION_LAW_TAIL_START.finditer(
        paragraph, entry.end()
    ):
        preceding = paragraph[: paragraph_candidate.start()].rstrip()
        separator = paragraph[len(preceding) : paragraph_candidate.start()]
        label = (paragraph_candidate.group("history_label") or "").lower()
        if (
            not label
            and preceding
            and "\n" not in separator
            and preceding[-1] not in ".!?"
        ):
            continue
        return candidate.start() + paragraph_candidate.start()
    return candidate.start() + paragraph_break.end() if paragraph_break else None


def _legacy_strip(source_text: str) -> str:
    terminal_parenthetical = re.search(r"\s+\((?P<history>[^()]*)\)\s*$", source_text)
    if terminal_parenthetical is not None:
        history_entries = tuple(
            entry.strip()
            for entry in terminal_parenthetical.group("history").split(";")
        )
        if history_entries and all(
            sc._ALABAMA_TERMINAL_ACT_HISTORY_ENTRY.fullmatch(entry)
            or sc._ALABAMA_TERMINAL_CODE_HISTORY_ENTRY.fullmatch(entry)
            for entry in history_entries
        ):
            return source_text[: terminal_parenthetical.start()].rstrip()

    retained_body_end = 0
    for candidate in sc._SESSION_LAW_TAIL_START.finditer(source_text):
        if candidate.start() < retained_body_end:
            continue
        prefix = source_text[: candidate.start()]
        preceding = prefix.rstrip()
        separator = prefix[len(preceding) :]
        explicitly_labeled = candidate.group("history_label") is not None
        strong_layout = not preceding or "\n" in separator
        if not explicitly_labeled and not strong_layout and preceding[-1] not in ".!?":
            continue
        entries = _legacy_validated_entries(source_text[candidate.start() :])
        if entries is None:
            resume_offset = _legacy_resume_offset(source_text, candidate)
            if resume_offset is not None:
                retained_body_end = resume_offset
                continue
            return source_text
        if (
            not explicitly_labeled
            and not strong_layout
            and (
                len(entries) < 2 or not any(entry.group("action") for entry in entries)
            )
        ):
            continue
        return prefix.rstrip()
    return source_text


# --- adversarial inputs from the issue ---------------------------------------


def _unlabeled_chain(n: int) -> str:
    """``1901, c. 1. 1902, c. 2. …`` after a sentence: every entry is a candidate."""

    chain = " ".join(f"{1900 + index % 100}, c. {index}." for index in range(n))
    return f"The credit is $40. {chain}"


def _citation_led_paragraphs(n: int) -> str:
    return "\n\n".join(
        f"Laws {1900 + index % 100}, c. {index} The amount is {index} dollars."
        for index in range(n)
    )


def _citation_led_single_paragraph(n: int) -> str:
    return " ".join(
        f"Laws {1900 + index % 100}, c. {index} The amount is {index} dollars."
        for index in range(n)
    )


def _measured_work(strip, text: str, monkeypatch) -> tuple[str, int]:
    meter = {"work": 0}
    whitespace_run_start = sc._whitespace_run_start

    def counted_whitespace_run_start(text, end, floor=0):
        start = whitespace_run_start(text, end, floor)
        meter["work"] += end - start + 1
        return start

    with monkeypatch.context() as patch:
        for name in METERED_PATTERNS:
            patch.setattr(sc, name, _CountingPattern(getattr(sc, name), meter))
        patch.setattr(sc, "_whitespace_run_start", counted_whitespace_run_start)
        result = strip(text)
    return result, meter["work"]


# Measured on this head: 4.1-5.7 reads per character on the inputs below;
# main's per-candidate re-normalization reads 500-1,000 per character.
WORK_PER_CHARACTER = 8

ADVERSARIAL = (
    pytest.param(_unlabeled_chain, id="unlabeled-chain"),
    pytest.param(_citation_led_paragraphs, id="citation-led-paragraphs"),
    pytest.param(_citation_led_single_paragraph, id="citation-led-single-paragraph"),
)


@pytest.mark.parametrize("build", ADVERSARIAL)
def test_strip_reads_each_character_a_bounded_number_of_times(build, monkeypatch):
    per_character = {}
    for size in (250, 1_000, 4_000):
        text = build(size)
        _result, work = _measured_work(
            sc._strip_terminal_session_law_history, text, monkeypatch
        )
        per_character[size] = work / len(text)
        # A constant number of reads per character: one normalization pass, one
        # paragraph scan, one candidate scan, and each candidate's own entry,
        # continuation and preceding whitespace.
        assert work <= WORK_PER_CHARACTER * len(text), (size, len(text), work)
    # Linear work keeps reads per character flat as the input grows fourfold.
    assert per_character[4_000] <= 1.1 * per_character[1_000], per_character


@pytest.mark.parametrize("build", ADVERSARIAL)
def test_work_meter_catches_the_quadratic_path(build, monkeypatch):
    """The same meter and bound fail main's per-candidate re-normalization."""

    text = build(1_000)
    _result, work = _measured_work(_legacy_strip, text, monkeypatch)

    assert work > WORK_PER_CHARACTER * len(text) * 50, (len(text), work)


class _CountingList(list):
    """A list that counts element reads, whether by index or by iteration.

    `bisect` reads a list subclass through ``__getitem__``; a linear scan reads
    it through ``__iter__``. Both go to the same counter.
    """

    def __init__(self, items, meter: dict[str, int]):
        super().__init__(items)
        self._meter = meter

    def __getitem__(self, index):
        self._meter["reads"] += 1
        return super().__getitem__(index)

    def __iter__(self):
        for item in super().__iter__():
            self._meter["reads"] += 1
            yield item


# The index lists `_SessionLawSuffixIndex.summary` searches or reads by index.
SEARCHED_INDEX_LISTS = (
    "_separator_starts",
    "_separator_ends",
    "_shifts",
    "_semicolons",
)


def _index_reads_per_lookup(text: str, monkeypatch) -> tuple[float, int]:
    """Element reads of the searched index lists per `summary` call.

    The regex meter cannot see Python-level searches, so a linear scan in
    place of `bisect` would be quadratic without failing the bound above.
    This counts reads, never time.
    """

    meter = {"reads": 0, "lookups": 0, "longest": 0}
    original_init = sc._SessionLawSuffixIndex.__init__
    original_summary = sc._SessionLawSuffixIndex.summary

    def counting_init(self, text):
        original_init(self, text)
        for name in SEARCHED_INDEX_LISTS:
            items = getattr(self, name)
            meter["longest"] = max(meter["longest"], len(items))
            setattr(self, name, _CountingList(items, meter))

    def counting_summary(self, start):
        meter["lookups"] += 1
        return original_summary(self, start)

    with monkeypatch.context() as patch:
        patch.setattr(sc._SessionLawSuffixIndex, "__init__", counting_init)
        patch.setattr(sc._SessionLawSuffixIndex, "summary", counting_summary)
        sc._strip_terminal_session_law_history(text)
    assert meter["lookups"], "the input must reach the suffix index"
    return meter["reads"] / meter["lookups"], meter["longest"]


@pytest.mark.parametrize("build", ADVERSARIAL)
def test_suffix_index_lookups_read_logarithmically_many_entries(build, monkeypatch):
    """Each `summary` call reads O(log n) index entries: two binary searches
    plus a constant number of direct reads."""

    for size in (1_000, 4_000):
        per_lookup, longest = _index_reads_per_lookup(build(size), monkeypatch)
        bound = 2 * (math.log2(longest + 1) + 1) + 8
        assert per_lookup <= bound, (size, longest, per_lookup, bound)


@pytest.mark.parametrize("build", ADVERSARIAL)
def test_adversarial_inputs_match_main(build):
    text = build(300)

    assert sc._strip_terminal_session_law_history(text) == _legacy_strip(text)


# --- differential --------------------------------------------------------------

_TOKENS = st.sampled_from(
    [
        "Laws ",
        "L. ",
        "P.L. ",
        "History: ",
        "Source: ",
        "Amended by ",
        "Added by ",
        "repealed ",
        "1901, c. 1",
        "2019, cc. 17, 18",
        "2021, Sp. Sess. I, c. 3",
        ", § 4",
        ", s. 12",
        ", eff. Jan. 1, 1992",
        ", eff. Feb. 30, 1992",
        ", operative July 1, 2020",
        ". ",
        "; ",
        ".",
        " ",
        "\n",
        "\n\n",
        " \n \n",
        "The amount is $40",
        "65 dollars",
        "(",
        '"',
        "unknown",
        "2024",
        "c. ",
    ]
)


@settings(max_examples=400, deadline=None)
@given(st.lists(_TOKENS, max_size=40).map("".join))
def test_property_strip_matches_main(text):
    assert sc._strip_terminal_session_law_history(text) == _legacy_strip(text)


def _fixture_texts():
    return [
        pytest.param(path, id=str(path.relative_to(FIXTURES)))
        for path in sorted(FIXTURES.rglob("*.txt"))
        if path.stat().st_size < 2_000_000
    ]


@pytest.mark.parametrize("path", _fixture_texts())
def test_differential_fixture_texts_match_main(path):
    text = path.read_text(errors="replace")

    assert sc._strip_terminal_session_law_history(text) == _legacy_strip(text)
