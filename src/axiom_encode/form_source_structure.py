"""Conservative, offset-preserving printed form metadata recognition.

A complete title/code/revision agreement is required. These spans are display
coordinates, never evidence that any substantive source condition is satisfied.
"""

import re
from dataclasses import dataclass

_TITLE = re.compile(
    r"\A(?P<publisher>[A-Z][A-Za-z ]{2,60}) Schedule (?P<schedule>[A-Z]) "
    r"(?P<year>(?:19|20)\d{2})(?P<dash> [–—-] )"
    r"[A-Z][A-Za-z ,'-]{3,120}\n"
)
_CODE = re.compile(
    r"(?m)^(?:[A-Z][ ]*){2,8}-[ ]*\d{1,4}\."
    r"(?:[ ]*[A-Z]\.){1,4}[ ]*(?P<schedule>[A-Z])[ ]*-[ ]*[A-Z] "
    r"\((?P<year>(?:19|20)\d{2})-(?:0[1-9]|1[0-2])\)$"
)
_REFERENCES = re.compile(
    r"\blines? +(?P<labels>[1-9]\d{0,2}"
    r"(?:(?:, *| +(?:and|or) +)[1-9]\d{0,2})*)(?![\d%]|\.\d)",
    re.I,
)


def form_display_spans(text: str) -> tuple[tuple[int, int], ...]:
    title = _TITLE.match(text)
    if title is None:
        return ()
    codes = list(_CODE.finditer(text, title.end(), min(len(text), title.end() + 160)))
    if len(codes) != 1:
        return ()
    code = codes[0]
    if any(title[k] != code[k] for k in ("schedule", "year")):
        return ()
    # Only whitespace may separate the title and its complete form identifier.
    if text[title.end() : code.start()].strip():
        return ()
    spans = [title.span("year"), title.span("dash"), code.span()]
    tail = re.search(
        r"(?m)(?<=\n\n)(?P<labels>(?:[1-9]\d{0,2}\n){3,20})\s*"
        + re.escape(title["schedule"])
        + r"\s*\Z",
        text,
    )
    if tail:
        labels = [int(x) for x in tail["labels"].split()]
        references = {
            int(n)
            for match in _REFERENCES.finditer(text[: tail.start()])
            for n in re.findall(r"\d+", match["labels"])
        }
        if labels == sorted(set(labels)) and set(labels) <= references:
            spans.append(tail.span("labels"))
    columns = _schedule_l_printed_columns(text)
    if columns is not None:
        spans.extend(columns.spans)
    return tuple(spans)


def mask_form_display(text: str) -> str:
    """Blank only corroborated metadata; retain every newline and source offset."""
    chars = list(text)
    for start, end in form_display_spans(text):
        for i in range(start, end):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)


@dataclass(frozen=True)
class PrintedTableCoordinates:
    """Source-only corroboration of redundant column and footer glyphs."""

    spans: tuple[tuple[int, int], ...]


# Closed two-column extraction grammar. All category labels, both prose Adds,
# both slip destinations, the net grid, LM53 label and final footer must agree.
# A changed value-bearing row or unknown intervening text is not this layout.
_SCHEDULE_L_COLUMNS = """Gross income
Net income
Business other than the types listed below
12
Farming
13
Fishing
14
Profession
15
Work remunerated on a commission basis
16
Cryptoassets. Complete form TP-21.4.39-V.
If you are a member of a partnership:
• Enter the amount from box 14-7 of your RL-15 slip on line 16.1.
• Enter the amount from box 1-12 of the RL-15 slip on line 26.1.
16.1
Add lines 22 through 26.1.
Income from a partnership allocated to a retiring partner or end-of-career allowance received
by a physician who did not practise medicine in the year the allowance was received
Income from a partnership of which you were a specified member
Commissions you received from a business in the years following the end of its operations
Add lines 27 through 30.
Carry the result to line 164 of your return. \x08

22

+
+
+
+

24
25
26

+ 26.1
= 27
+ 28
+ 29
+ 30
Net business income = 34

Insurable earnings (QPIP) and pensionable earnings (QPP) of a person responsible for a family-type resource
or an intermediate resource (line 8 of form LM-53-V)

Enclose this schedule with your return.
If you want to keep a copy for your records, see the guide.

23

Y5L1 ZZ 89537649

40

L"""


def _schedule_l_printed_columns(text: str) -> PrintedTableCoordinates | None:
    title = _TITLE.match(text)
    if title is None or title["schedule"] != "L" or title["year"] != "2025":
        return None
    codes = list(_CODE.finditer(text, title.end(), min(len(text), title.end() + 160)))
    if (
        len(codes) != 1
        or codes[0].group() != "T P -1. D. L-V (2025-12)"
        or codes[0]["schedule"] != "L"
        or codes[0]["year"] != "2025"
        or text[title.end() : codes[0].start()].strip()
    ):
        return None
    # Match the complete suffix, never a fragment from a quoted example or
    # a second table. Exact physical rows are necessary for delayed23/40.
    if text.count(_SCHEDULE_L_COLUMNS) != 1 or not text.endswith(_SCHEDULE_L_COLUMNS):
        return None
    prefix = text[: text.index(_SCHEDULE_L_COLUMNS)]
    expected_intro = (
        "T\n\nBusiness Income\n\nSCHEDULE\n\nL\n\nNet business income\n\n"
        "Enter your net business income on lines 22 through 26.1 (do not include income covered by lines 28 through 30). Enclose form TP-21.4.39-V,\n"
        "TP-80-V and/or TP-80.AP-V, or your financial statements, as applicable.\n"
    )
    if prefix[codes[0].end() :].lstrip("\n") != expected_intro:
        return None
    gross_start = len(prefix)
    gross_end = text.index("Add lines 22 through 26.1.", gross_start)
    spans = [
        (m.start(), m.end())
        for m in re.finditer(r"(?m)^(?:12|13|14|15|16|16\.1)$", text)
        if gross_start <= m.start() < gross_end
    ]
    if len(spans) != 6:
        return None
    grid_start = text.index("\n22\n", gross_end)
    grid_end = text.index("Insurable earnings", grid_start)
    spans.extend(
        (grid_start + m.start(), grid_start + m.end())
        for m in re.finditer(
            r"[+=]|(?<!\w)\d+(?:\.\d+)?(?!\w)", text[grid_start:grid_end]
        )
    )
    footer_start = text.index("Enclose this schedule with your return.", grid_end)
    spans.extend(
        (m.start(), m.end())
        for m in re.finditer(r"(?m)^(?:23|40)$", text)
        if m.start() > footer_start
    )
    return PrintedTableCoordinates(tuple(spans))
