"""Closed printed-row boundaries, not evidence of satisfied source conditions.

All returned spans refer to the original text. Inherited guards and an explicit
within-row if/otherwise stay atomic; unknown dependent rows refuse recognition.
"""

import re
from dataclasses import dataclass

_TITLE = re.compile(r"\AStep [1-9]\d? (?P<dash>-) [A-Z][A-Za-z -]{3,80}\r?\n")
_ROW = re.compile(
    r"(?m)^Line (?P<row>[1-9]\d{0,4})(?: and (?:line )?(?P<field>[1-9]\d{4}))?:"
    r"[ \t]+(?P<caption>[A-Z][^\r\n]{1,500})$"
)
_INHERITED = re.compile(
    r"(?m)^[A-Z][A-Za-z -]{3,80}:\r?\n\r?\n"
    r"Complete the chart for line (?P<field>[1-9]\d{4}) using your Federal Worksheet "
    r"if one or more of the following apply:\r?\n"
    r"(?:\r?\n- [^\r\n]+\r?\n){2,6}\r?\n"
    r"Line (?P<row>[1-9]\d{0,2}) and line (?P=field): "
    r'If not, enter "0" on line (?P=field)\. \^'
)
_CONDITIONAL = re.compile(
    r"[^;\r\n]+\(enter the amount from line [1-9]\d{4} "
    r"if you did not enter an amount on line [1-9]\d{4}; "
    r"otherwise, use Federal Worksheet\) \^"
)

_ADD_ROW = re.compile(
    r"(?P<operation>Add lines (?P<a>[1-9]\d{0,2}) to (?P<b>[1-9]\d{0,2})"
    r"(?: \(see line [1-9]\d{4} in Step [1-9]\d?\))?\.)"
    r"(?: Net self-employment income)? = \^"
)


@dataclass(frozen=True)
class PrintedRowStructure:
    boundaries: tuple[int, ...]
    atomic_spans: tuple[tuple[int, int], ...]
    heading_dash: tuple[int, int]
    additions: tuple[tuple[int, int], ...]


def recognize_printed_rows(text: str) -> PrintedRowStructure | None:
    title = _TITLE.match(text)
    if title is None or len(re.findall(r"(?m)^Step \d+ - ", text)) != 1:
        return None
    if text.count('"') % 2 or text.count("“") != text.count("”"):
        return None
    quoted = [m.span() for m in re.finditer(r'"[^\"]*"|“[^”]*”', text)]
    rows = list(_ROW.finditer(text))
    if len(rows) != len(re.findall(r"(?m)^Line ", text)):
        return None
    labels = [int(m["row"]) for m in rows]
    display = [n for n in labels if n < 1000]
    fields = [int(m["field"]) for m in rows if m["field"]]
    if (
        len(display) < 3
        or display != list(range(display[0], display[-1] + 1))
        or len(labels) != len(set(labels))
        or len(fields) != len(set(fields))
        or not fields
        or set(fields) & set(labels)
        or any(
            a < m.start("caption") and m.start() < b for m in rows for a, b in quoted
        )
    ):
        return None
    if not any(
        int(m[1]) < int(m[2]) and set(range(int(m[1]), int(m[2]) + 1)) <= set(display)
        for m in re.finditer(r"\bAdd lines ([1-9]\d{0,2}) to ([1-9]\d{0,2})\.", text)
    ):
        return None
    additions = []
    for row in rows:
        add = _ADD_ROW.fullmatch(row["caption"])
        if (
            add
            and int(add["a"]) < int(add["b"]) < int(row["row"])
            and set(range(int(add["a"]), int(add["b"]) + 1)) <= set(display)
        ):
            additions.append(
                (
                    row.start("caption") + add.start("operation"),
                    row.start("caption") + add.end("operation"),
                )
            )
    inherited = list(_INHERITED.finditer(text))
    atomic = []
    for row in rows:
        if re.match(r"(?:If|Otherwise|Then)\b", row["caption"]):
            groups = [
                m
                for m in inherited
                if m["row"] == row["row"]
                and m["field"] == row["field"]
                and m.end() == row.end()
            ]
            if len(groups) != 1:
                return None
            atomic.append(groups[0].span())
        elif _CONDITIONAL.fullmatch(row["caption"]):
            atomic.append(row.span())
    # The exact group may not consume another printed row or a quoted example.
    for start, end in atomic:
        if sum(start <= m.start() < end for m in rows) != 1:
            return None
    boundaries = {title.start(), title.end(), *(i for m in rows for i in m.span())}
    boundaries.update(i for pair in atomic for i in pair)
    return PrintedRowStructure(
        tuple(sorted(boundaries)), tuple(atomic), title.span("dash"), tuple(additions)
    )


def mask_printed_heading_dash(text: str) -> str:
    evidence = recognize_printed_rows(text)
    if evidence is None:
        return text
    a, b = evidence.heading_dash
    return text[:a] + " " * (b - a) + text[b:]
