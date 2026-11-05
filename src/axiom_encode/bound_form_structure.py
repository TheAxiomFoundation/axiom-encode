"""Source-only ownership of one printed form, never completeness evidence.

These raw spans describe a closed transcription layout. They neither mask
numbers nor establish dates, arithmetic, eligibility, or filing compliance.
A consumer must bind them to its authoritative body and independently prove
any proposed use of a header, dimension or row.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

Span = tuple[int, int]
_HEADER = re.compile(
    r"\A\ufeff?(?P<code>\d{4,5}-[A-Z]{1,4})\r?\n\r?\n"
    r"Protected [A-Z] when completed\r?\n\r?\n"
    r"(?P<year>(?:19|20)\d{2})\r?\n\r?\n"
    r"Form (?P<form>[A-Z]{1,4}\d{1,4})\r?\n\r?\n"
    r"(?P<title>[A-Z][A-Za-z ,'-]{3,120})(?=\r?\n\r?\n)"
)
_NOTE = re.compile(
    r"(?m)^\*\*\* Transcriber's Note: In print, the table is set-up as "
    r"(?P<columns>[1-9]\d?)\s+columns with (?P<entries>[1-9]\d{0,2})\s+"
    r"entry lines for each column\. \*\*\*\r?$"
)
_ROW = re.compile(
    r"(?m)^Line (?P<row>[1-9]\d{0,3})"
    r"(?: and line (?P<field>[1-9]\d{4}))?: (?P<caption>[^\r\n]+)"
)
_FIELD = re.compile(r"(?m)^(?P<label>[A-Za-z][A-Za-z ()'/-]{1,160}) \^\r?$")
_CITATION = re.compile(r"[a-z0-9][a-z0-9-]*(?:/[a-z0-9][a-z0-9-]*)+")


@dataclass(frozen=True)
class PrintedFormHeader:
    span: Span
    year_span: Span
    year: int
    code: str
    form: str
    title: str


@dataclass(frozen=True)
class TranscribedEntryTable:
    note_span: Span
    column_count_span: Span
    entry_count_span: Span
    column_count: int
    entry_count: int
    field_spans: tuple[Span, ...]
    total_row: int


@dataclass(frozen=True)
class WrappedFormRow:
    number: int
    field_number: int | None
    span: Span
    caption_span: Span


@dataclass(frozen=True)
class BoundFormStructure:
    source: str
    citation: str
    source_sha256: str
    header: PrintedFormHeader
    table: TranscribedEntryTable
    rows: tuple[WrappedFormRow, ...]
    # Includes preamble, subsection headings, separated qualifications and tail.
    # A consumer must retain these in ordinary source/condition analysis.
    unowned_spans: tuple[Span, ...]

    def bound(self, source: str, citation: str) -> bool:
        """Reject changed bodies/citations and forged or clipped span records."""
        return (
            source == self.source
            and citation == self.citation
            and recognize_bound_form_structure(source, citation) == self
        )


def recognize_bound_form_structure(
    source: str, citation: str
) -> BoundFormStructure | None:
    """Recognize complete physical ownership; unknown layouts fail closed."""
    if (
        not isinstance(source, str)
        or not isinstance(citation, str)
        or not _CITATION.fullmatch(citation)
        or len(source) > 100_000
        or any(mark in source for mark in ('"', "“", "”"))
    ):
        return None
    header = _HEADER.match(source)
    notes = tuple(_NOTE.finditer(source))
    starts = tuple(_ROW.finditer(source))
    if (
        header is None
        or len(notes) != 1
        or len(re.findall(r"(?m)^\*\*\* Transcriber's Note:", source)) != 1
        or len(re.findall(r"(?m)^Form [A-Z]+\d+\r?$", source)) != 1
        or len(starts) < 3
        or len(starts) != len(re.findall(r"(?m)^Line ", source))
    ):
        return None
    note = notes[0]
    if not header.end() < note.start() < note.end() < starts[0].start():
        return None
    numbers = tuple(int(m["row"]) for m in starts)
    fields = tuple(int(m["field"]) for m in starts if m["field"] is not None)
    if (
        numbers != tuple(range(numbers[0], numbers[-1] + 1))
        or len(fields) != len(set(fields))
        or set(numbers) & set(fields)
    ):
        return None
    field_text = source[note.end() : starts[0].start()]
    labels = tuple(_FIELD.finditer(field_text))
    remaining = _FIELD.sub("", field_text)
    if (
        remaining.strip()
        or len(labels) != int(note["columns"])
        or len({m["label"].casefold() for m in labels}) != len(labels)
    ):
        return None
    rows = []
    for index, start in enumerate(starts):
        limit = starts[index + 1].start() if index + 1 < len(starts) else len(source)
        blank = re.search(r"\r?\n[ \t]*\r?\n", source[start.end() : limit])
        end = start.end() + blank.start() if blank else limit
        while end > start.end() and source[end - 1] in "\r\n":
            end -= 1
        rows.append(
            WrappedFormRow(
                int(start["row"]),
                int(start["field"]) if start["field"] else None,
                (start.start(), end),
                (start.start("caption"), end),
            )
        )
    # The entry labels are owned by a full table-total row, not a nearby tax
    # amount or an arbitrary first calculation. Other rows remain uninterpreted.
    total = source[slice(*rows[0].caption_span)]
    if not re.fullmatch(
        r"Total amount paid from all receipts included in the table above\s+"
        r"Total\s+eligible expenses = \^",
        total,
    ):
        return None
    field_spans = tuple((note.end() + m.start(), note.end() + m.end()) for m in labels)
    owned = sorted((header.span(), note.span(), *field_spans, *(r.span for r in rows)))
    gaps = []
    previous = 0
    for start, end in (*owned, (len(source), len(source))):
        if start < previous:
            return None
        if source[previous:start].strip():
            gaps.append((previous, start))
        previous = end
    return BoundFormStructure(
        source,
        citation,
        hashlib.sha256(source.encode()).hexdigest(),
        PrintedFormHeader(
            header.span(),
            header.span("year"),
            int(header["year"]),
            header["code"],
            header["form"],
            header["title"],
        ),
        TranscribedEntryTable(
            note.span(),
            note.span("columns"),
            note.span("entries"),
            int(note["columns"]),
            int(note["entries"]),
            field_spans,
            rows[0].number,
        ),
        tuple(rows),
        tuple(gaps),
    )
