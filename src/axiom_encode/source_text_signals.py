"""Deterministic signals read from corpus source text (stdlib only).

Shared by the SNAP queue builder (``scripts/snap_dispatch_queue.py``) and the
encoder's pre-encode gate, so both judge a source unit the same way.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Tables of contents are dense with dot leaders ("Introduction....... 12");
# policy text almost never has them. In the Oregon notebook, contents pages
# have 24+ leader runs and policy pages 0-2.
TOC_LEADER = re.compile(r"\.{8,}")
TOC_MIN_LEADERS = 10


def is_table_of_contents(text: str) -> bool:
    """Return whether ``text`` reads as a table of contents page."""

    return len(TOC_LEADER.findall(text)) >= TOC_MIN_LEADERS


# One contents entry: a title of up to 160 characters, a dot-leader run, and a
# page reference (a section-style number such as "12", "4-3", "M0310.000" or
# "42(h)", or a roman numeral). Leaders in a fill-in form ("on the ..........
# day of") are not followed by one, and an amount with a thousands separator
# ("1,355,831" in an appropriations table) is not one.
TOC_ENTRY = re.compile(
    r"(?:[^.\n]|\.(?!\.{7})){0,160}\.{8,}[ \t]*"
    r"(?:[A-Za-z]{0,3}[\-–.]?\d[\w\-–.:()]{0,12}|[ivxlcdmIVXLCDM]{1,6})"
    r"(?![\w,])"
)
# A contents page names itself ("Table of Contents", "Contents").
CONTENTS_MARKER = re.compile(r"\bcontents\b", re.IGNORECASE)


def text_outside_contents_entries(text: str) -> float:
    """Return the share of non-space characters outside contents entries.

    A contents page is almost all entries. A long chapter that opens with its
    own table of contents keeps most of its characters outside them.
    """

    total = len(re.sub(r"\s", "", text))
    if not total:
        return 0.0
    rest = len(re.sub(r"\s", "", TOC_ENTRY.sub(" ", text)))
    return rest / total


_SENTENCE_BREAK = re.compile(r"(?<=[.:;!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
# A prose sentence: at least 8 words, most of them lower case. Contents
# headers and running heads are title case or capitals.
PROSE_MIN_WORDS = 8


def prose_outside_contents_entries(text: str) -> str | None:
    """Return the first prose sentence outside contents entries, if any."""

    for segment in TOC_ENTRY.split(text):
        for sentence in _SENTENCE_BREAK.split(segment):
            words = _WORD.findall(sentence)
            if len(words) < PROSE_MIN_WORDS:
                continue
            lower = sum(1 for word in words if word[0].islower())
            if lower * 2 >= len(words):
                return " ".join(sentence.split())
    return None


# Words and signs that make a rule: a modal, a predicate verb, a negation, an
# amount or a rate. Contents titles and running heads ("Chapter 11 Benefit
# Repayment", "Table of Contents (cont'd) Page 3") carry none of them.
_RULE_SIGNAL = re.compile(
    r"\b(?:shall|must|may|will|would|can|cannot|could|should|is|are|was|were|"
    r"be|been|being|has|have|had|does|do|did|means|mean|includes?|including|"
    r"requires?|required|applies|apply|eligible|ineligible|entitled|counts?|"
    r"counted|excluded?|allowed|allows?|receives?|reports?|reported|equals?|"
    r"exceeds?|not|no)\b|[$%]",
    re.IGNORECASE,
)


def rule_signal_outside_contents_entries(
    text: str, labels: Sequence[str] = ()
) -> str | None:
    """Return text outside contents entries that may state a rule, if any.

    A page is judged structural only when everything outside its entries is
    positively explained: no prose sentence, no rule word, amount or rate, and
    no word that is not contents-page furniture, a roman page number, or a
    word that also occurs in an entry or in the row's own labels (a running
    head, a chapter title listed below). Capitalization proves nothing:
    "SUBMIT APPLICATIONS WITHIN 30 DAYS." is a rule.
    """

    prose = prose_outside_contents_entries(text)
    if prose is not None:
        return prose
    segments = TOC_ENTRY.split(text)
    for segment in segments:
        match = _RULE_SIGNAL.search(segment)
        if match is not None:
            start = max(0, match.start() - 60)
            return " ".join(segment[start : match.end() + 60].split())
    entries = " ".join(match.group(0) for match in TOC_ENTRY.finditer(text))
    known = {word.casefold() for word in _WORD.findall(entries)}
    for label in labels:
        known.update(word.casefold() for word in _WORD.findall(label))
    known.update(_CONTENTS_FURNITURE_WORDS)
    for segment in segments:
        for word in _WORD.findall(segment):
            folded = word.casefold()
            if folded in known or re.fullmatch(r"[ivxlcdm]{1,6}", folded):
                continue
            return word
    # A number outside the entries must be a locator: a page or chapter
    # number after its unit word, the row's own page number (from its labels)
    # where a running head puts it, at the start of a line, or a revision
    # date. "Period: 12 Months." is a limit, even on page 12.
    label_numbers = {number for label in labels for number in _NUMBER.findall(label)}
    for segment in segments:
        for match in _NUMBER.finditer(segment):
            number = match.group(0).rstrip(".:-–")
            if _REVISION_DATE.fullmatch(number):
                continue
            line_start = segment.rfind("\n", 0, match.start()) + 1
            if (
                number in label_numbers
                and not segment[line_start : match.start()].strip()
            ):
                continue
            before = _WORD.findall(segment[: match.start()])
            if before and before[-1].casefold() in _LOCATOR_UNIT_WORDS:
                continue
            return " ".join(
                segment[max(0, match.start() - 40) : match.end() + 20].split()
            )
    return None


_NUMBER = re.compile(r"\d[\d./\-–:]*")
_REVISION_DATE = re.compile(r"\d{1,2}/(?:\d{1,2}/)?\d{2,4}")
_LOCATOR_UNIT_WORDS = frozenset(
    {"appendix", "article", "chapter", "page", "part", "section", "subpart", "title"}
)


# Words a contents page's own furniture uses ("Table of Contents (cont'd)
# Page 3", "Chapter 11").
_CONTENTS_FURNITURE_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "appendix",
        "article",
        "as",
        "at",
        "by",
        "chapter",
        "cont'd",
        "cont’d",
        "content",
        "contents",
        "continued",
        "for",
        "from",
        "in",
        "index",
        "manual",
        "of",
        "on",
        "or",
        "page",
        "part",
        "section",
        "subpart",
        "table",
        "the",
        "title",
        "to",
        "toc",
        "with",
    }
)
