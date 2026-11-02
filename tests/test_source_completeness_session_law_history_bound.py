"""The terminal session-law history strip does linear work (#1828).

The strip used to re-normalize and re-split the whole remaining suffix for each
candidate history start, so its work was text length times candidates. These
tests count the characters the session-law normalizer and paragraph scanner
read, not elapsed time, and keep main's implementation as a reference: the
fixed strip must return exactly what main returned on every input.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

FIXTURES = Path(__file__).parent / "fixtures"


class _CountingPattern:
    """A compiled pattern that adds the span it is asked to read to ``work``."""

    def __init__(self, pattern: re.Pattern[str], meter: dict[str, int]):
        self._pattern = pattern
        self._meter = meter
        self.pattern = pattern.pattern
        self.flags = pattern.flags

    def _count(self, string: str, pos: int = 0, endpos: int | None = None) -> None:
        end = len(string) if endpos is None else min(endpos, len(string))
        self._meter["work"] += max(0, end - pos)

    def sub(self, repl, string, count=0):
        self._count(string)
        return self._pattern.sub(repl, string, count)

    def search(self, string, pos=0, endpos=None):
        self._count(string, pos, endpos)
        args = (string, pos) if endpos is None else (string, pos, endpos)
        return self._pattern.search(*args)

    def finditer(self, string, pos=0, endpos=None):
        self._count(string, pos, endpos)
        args = (string, pos) if endpos is None else (string, pos, endpos)
        return self._pattern.finditer(*args)


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
    with monkeypatch.context() as patch:
        patch.setattr(
            sc,
            "_SESSION_LAW_SENTENCE_SEPARATOR",
            _CountingPattern(sc._SESSION_LAW_SENTENCE_SEPARATOR, meter),
        )
        patch.setattr(
            sc,
            "_SESSION_LAW_PARAGRAPH_BREAK",
            _CountingPattern(sc._SESSION_LAW_PARAGRAPH_BREAK, meter),
        )
        result = strip(text)
    return result, meter["work"]


WORK_PER_CHARACTER = 3

ADVERSARIAL = (
    pytest.param(_unlabeled_chain, id="unlabeled-chain"),
    pytest.param(_citation_led_paragraphs, id="citation-led-paragraphs"),
    pytest.param(_citation_led_single_paragraph, id="citation-led-single-paragraph"),
)


@pytest.mark.parametrize("build", ADVERSARIAL)
def test_strip_reads_each_character_a_bounded_number_of_times(build, monkeypatch):
    for size in (250, 1_000, 4_000):
        text = build(size)
        _result, work = _measured_work(
            sc._strip_terminal_session_law_history, text, monkeypatch
        )
        # One normalization pass, one paragraph scan, and each candidate's own
        # first entry: a constant number of reads per character.
        assert work <= WORK_PER_CHARACTER * len(text), (size, len(text), work)


@pytest.mark.parametrize("build", ADVERSARIAL)
def test_work_meter_catches_the_quadratic_path(build, monkeypatch):
    """The same meter and bound fail main's per-candidate re-normalization."""

    text = build(1_000)
    _result, work = _measured_work(_legacy_strip, text, monkeypatch)

    assert work > WORK_PER_CHARACTER * len(text) * 50, (len(text), work)


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
