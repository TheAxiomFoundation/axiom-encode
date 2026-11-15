"""Raw Schedule layout records, with no execution or condition authority.

Detached reporting fields and conditional column captions remain unowned text.
Recognition does not admit a snapshot, certify arithmetic, or mask source cues.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

Span = tuple[int, int]
_HEADER = re.compile(
    r"\ufeff?(?P<code>\d{4,5}-[A-Z]{1,4}\d{1,3})\r?\n\r?\n"
    r"Protected [A-Z] (?P<cue>when) completed\r?\n\r?\n"
    r"(?P<year>(?:19|20)\d{2})\r?\n\r?\n"
    r"Schedule (?P<form>[A-Z]{1,4}\([A-Z]\d{1,3}\))\r?\n\r?\n"
    r"(?P<title>[A-Z][A-Za-z ,'-]{3,120})(?=\r?\n\r?\n)"
)
_WRAPPER = re.compile(
    r"\ACRA Schedule (?P<form>[A-Z]{1,4}\([A-Z]\d{1,3}\)) "
    r"(?P<year>(?:19|20)\d{2}) (?P<title>[A-Z][A-Za-z ,'-]{3,120})\r?\n\r?\n"
)
_OPERATIVE = re.compile(
    r"\b(?:if|unless|except|provided|otherwise|when|plus|minus|less|"
    r"subtract|add|multiply|divide|maximum|minimum|deduct)\b",
    re.I,
)
_NOTE = re.compile(
    r"(?m)^\*\*\* Transcriber's Note: In print, the following table is set up in "
    r"(?P<columns>[1-9]\d?) columns\r?\n"
    r"with the possibility of (?P<entries>[1-9]\d{0,2}) entries\. \*\*\*\r?$"
)
_FIELD = re.compile(r"(?m)^(?P<label>[A-Za-z][A-Za-z ()'/-]{1,160}) \^\r?$")
_LINE = re.compile(
    r"(?m)^Line (?P<number>[1-9]\d{0,4})"
    r"(?: and line (?P<field>[1-9]\d{4}))?: (?P<caption>[^\r\n]+)"
)


@dataclass(frozen=True)
class ScheduleRow:
    number: int
    field_number: int | None
    span: Span
    caption_span: Span


@dataclass(frozen=True)
class BoundScheduleStructure:
    source: str
    citation: str
    source_sha256: str
    header_span: Span
    header_year_span: Span
    confidentiality_cue_span: Span
    wrapper_year_span: Span | None
    form: str
    year: int
    note_span: Span
    column_count_span: Span
    entry_count_span: Span
    entry_field_spans: tuple[Span, ...]
    rows: tuple[ScheduleRow, ...]
    reporting_fields: tuple[ScheduleRow, ...]
    # Deliberately includes reporting fields, column qualifiers and all prose.
    unowned_spans: tuple[Span, ...]

    def bound(self, source: str, citation: str) -> bool:
        return (
            source == self.source
            and citation == self.citation
            and recognize_bound_schedule_structure(source, citation) == self
        )

    def entered_table_caption_span(self) -> Span | None:
        """The closed entered-field heading block is not an arithmetic formula.

        Its applicability text is still ordinary condition-discovery input.
        Unknown columns and operative additions receive no classification.
        """
        expected = (
            "Date on sales slip or contract ^",
            "Supplier or contractor - Name ^",
            "Supplier or contractor - GST/HST Number (if applicable) ^",
            "Description ^",
            "Amount paid (including all applicable taxes) ^",
        )
        if (
            tuple(self.source[a:b].strip() for a, b in self.entry_field_spans)
            != expected
        ):
            return None
        start, end = self.entry_field_spans[0][0], self.entry_field_spans[-1][1]
        if self.source[start:end].replace("\r\n", "\n") != "\n".join(expected):
            return None
        return start, end


def recognize_bound_schedule_structure(
    source: str, citation: str
) -> BoundScheduleStructure | None:
    """Describe the complete raw layout; never infer legal obligations."""
    if (
        not isinstance(source, str)
        or len(source) > 100_000
        or not isinstance(citation, str)
        or not re.fullmatch(r"[a-z0-9-]+(?:/[a-z0-9-]+)+", citation)
    ):
        return None
    wrapper = _WRAPPER.match(source)
    header = _HEADER.match(source, wrapper.end() if wrapper else 0)
    if header is None or _OPERATIVE.search(header["title"]):
        return None
    if wrapper and any(wrapper[k] != header[k] for k in ("form", "year", "title")):
        return None
    # Printed code and schedule suffix must agree; the jurisdiction is not a whitelist.
    if header["code"].split("-")[-1] != header["form"].split("(")[-1][:-1]:
        return None
    if len(re.findall(r"(?m)^Schedule ", source)) != 1:
        return None
    notes = tuple(_NOTE.finditer(source))
    starts = tuple(_LINE.finditer(source))
    if (
        len(notes) != 1
        or len(re.findall(r"(?m)^\*\*\* Transcriber's Note:", source)) != 1
        or len(starts) != len(re.findall(r"(?m)^Line ", source))
        or not starts
    ):
        return None
    note = notes[0]
    if not header.end() < note.start() < note.end() < starts[0].start():
        return None
    text = source[note.end() : starts[0].start()]
    fields = tuple(_FIELD.finditer(text))
    if (
        _FIELD.sub("", text).strip()
        or len(fields) != int(note["columns"])
        or len({m["label"].casefold() for m in fields}) != len(fields)
    ):
        return None
    records = []
    for i, start in enumerate(starts):
        limit = starts[i + 1].start() if i + 1 < len(starts) else len(source)
        blank = re.search(r"\r?\n[ \t]*\r?\n", source[start.end() : limit])
        end = start.end() + blank.start() if blank else limit
        while end > start.end() and source[end - 1] in "\r\n":
            end -= 1
        records.append(
            ScheduleRow(
                int(start["number"]),
                int(start["field"]) if start["field"] else None,
                (start.start(), end),
                (start.start("caption"), end),
            )
        )
    rows = tuple(r for r in records if r.number < 10000)
    reporting = tuple(r for r in records if r.number >= 10000)
    if (
        len(rows) < 3
        or not reporting
        or tuple(r.number for r in rows) != tuple(range(1, len(rows) + 1))
        or tuple(records) != rows + reporting
        or len({r.number for r in reporting}) != len(reporting)
        or any(r.field_number is not None for r in reporting)
        or any(not source[slice(*r.caption_span)].endswith("^") for r in reporting)
    ):
        return None
    aliases = [r.field_number for r in rows if r.field_number is not None]
    if len(aliases) != len(set(aliases)) or set(aliases) & {
        r.number for r in reporting
    }:
        return None
    if not re.fullmatch(
        r"Total amount paid from all sales slips and contracts included in the\s+"
        r"table above Total eligible expenses = \^",
        source[slice(*rows[0].caption_span)],
    ):
        return None
    field_spans = tuple((note.end() + f.start(), note.end() + f.end()) for f in fields)
    # Column captions are descriptors, not owned obligations: 'if applicable' stays raw.
    owned = sorted((header.span(), note.span(), *(r.span for r in rows)))
    gaps = []
    previous = 0
    for start, end in (*owned, (len(source), len(source))):
        if start < previous:
            return None
        if source[previous:start].strip():
            gaps.append((previous, start))
        previous = end
    return BoundScheduleStructure(
        source,
        citation,
        hashlib.sha256(source.encode()).hexdigest(),
        header.span(),
        header.span("year"),
        header.span("cue"),
        wrapper.span("year") if wrapper else None,
        header["form"],
        int(header["year"]),
        note.span(),
        note.span("columns"),
        note.span("entries"),
        field_spans,
        rows,
        reporting,
        tuple(gaps),
    )
