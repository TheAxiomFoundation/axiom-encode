"""Numeric-only locator evidence, never formula or condition segmentation.

Complete slip referrals bind box labels to a printed slip identifier. Form
rows/footer additionally require a coherent printed step and unique row grid.
Only the locator digits are blanked; captions, amounts and operators survive.
"""

import re

_BOX_REFERRAL = (
    r"\bbox(?:es)?[ \t]+(?P<labels>\d{1,3}"
    r"(?:(?:,[ \t]+(?:(?:and|or)[ \t]+)?|[ \t]+(?:and|or)[ \t]+)\d{1,3})*)"
    r"[ \t]+of[ \t]+(?:all|the|your)[ \t]+"
    r"[A-Z]{1,4}\d{1,4}[A-Z]{0,4}(?:\([A-Z]{1,4}\))?"
    r"[ \t]+slips?"
)
_REFERRAL_END = r"(?:$|[ \t]*[).,;:\r\n]|[ \t]+or[ \t]+see[ \t]+Form\b)"
_SLIP_BOX = re.compile(
    _BOX_REFERRAL
    + r"(?="
    + _REFERRAL_END
    + r"|[ \t]+and[ \t]+"
    + _BOX_REFERRAL.replace("(?P<labels>", "(?:")
    + _REFERRAL_END
    + r")",
    re.I,
)
_STEP = re.compile(r"\AStep (?P<step>[1-9]\d?) - [A-Z][A-Za-z -]{3,80}\r?\n")
_ROW = re.compile(
    r"(?m)^Line (?P<row>[1-9]\d{0,4})(?: and (?:line )?(?P<field>[1-9]\d{4}))?:"
    r"[ \t]+(?P<caption>[A-Z][^\r\n]{1,500})$"
)
_FOOTER = re.compile(
    r"(?m)^(?P<code>\d{4,5}-[A-Z]{1,3}) \(Page (?P<page>[1-9]\d?) "
    r"of (?P<total>[1-9]\d?)\)[ \t]*(?:\r?\n)*\Z"
)

_PARENTHETICAL_REFERRAL = re.compile(
    r"(?<![()])\((?:complete Schedule (?P<schedule>[1-9]\d{0,2})|"
    r"see line (?P<line>[1-9]\d{4}) in Step (?P<step>[1-9]\d?))\)"
    r"(?=[ \t]*(?:\.[ \t]*)?(?:=[ \t]*)?\^[ \t]*$)"
)


def form_numeric_locator_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return exact spans only for complete, unambiguous locator evidence."""
    # Unbalanced quotation is not reliable structural evidence. Apostrophes
    # inside prose words are not quotation delimiters.
    if text.count('"') % 2 or text.count("“") != text.count("”"):
        return ()
    quoted = [m.span() for m in re.finditer(r'"[^\"]*"|“[^”]*”', text)]

    def unquoted(start: int, end: int) -> bool:
        return not any(a < end and start < b for a, b in quoted)

    spans = []
    for match in _SLIP_BOX.finditer(text):
        labels = list(re.finditer(r"\d+", match["labels"]))
        if len({int(x.group()) for x in labels}) != len(labels):
            continue
        if not unquoted(*match.span()):
            continue
        spans.extend(
            (match.start("labels") + x.start(), match.start("labels") + x.end())
            for x in labels
        )
    title = _STEP.match(text)
    if title is None or len(re.findall(r"(?m)^Step \d+ - ", text)) != 1:
        return tuple(spans)
    rows = list(_ROW.finditer(text))
    # No hidden/malformed or duplicated printed rows may corroborate a footer.
    if len(rows) != len(re.findall(r"(?m)^Line ", text)):
        return tuple(spans)
    labels = [int(m["row"]) for m in rows]
    display = [n for n in labels if n < 1000]
    fields = [m["field"] for m in rows if m["field"] is not None]
    if (
        len(display) < 3
        or display != list(range(display[0], display[-1] + 1))
        or len(labels) != len(set(labels))
        or len(fields) != len(set(fields))
        or {int(x) for x in fields} & set(labels)
        or not fields
        or any(not unquoted(m.start(), m.start("caption")) for m in rows)
    ):
        return tuple(spans)
    for row in rows:
        # A complete balanced referral is structural only at a printed entry
        # row's end. Amounts, arithmetic suffixes and nested examples refuse.
        for referral in _PARENTHETICAL_REFERRAL.finditer(row["caption"]):
            # The closed scope is one complete referral, not a nested or
            # partially delimited discussion containing a matching substring.
            if row["caption"].count("(") != 1 or row["caption"].count(")") != 1:
                continue
            start = row.start("caption") + referral.start()
            end = row.start("caption") + referral.end()
            if not unquoted(start, end):
                continue
            for group in ("schedule", "line", "step"):
                if referral[group] is not None:
                    a, b = referral.span(group)
                    spans.append((row.start("caption") + a, row.start("caption") + b))
        if row["field"] and int(row["row"]) < 1000:
            spans.append(row.span("field"))
    footers = list(_FOOTER.finditer(text))
    if len(footers) != 1 or len(re.findall(r"(?m)^\d{4,5}-[A-Z]+\b", text)) != 1:
        return tuple(spans)
    footer = footers[0]
    if not (int(title["step"]) < int(footer["page"]) <= int(footer["total"])):
        return tuple(spans)
    # An actual add-row links the displayed coordinates into a calculation;
    # merely listing numeric rows does not authenticate a form footer.
    adds = list(re.finditer(r"\bAdd lines ([1-9]\d{0,2}) to ([1-9]\d{0,2})\.", text))
    if not any(
        int(m[1]) < int(m[2])
        and set(range(int(m[1]), int(m[2]) + 1)) <= set(display)
        and unquoted(*m.span())
        for m in adds
    ) or not unquoted(*footer.span()):
        return tuple(spans)
    spans.extend(
        [
            (
                footer.start("code"),
                footer.start("code") + len(footer["code"].split("-")[0]),
            ),
            footer.span("page"),
            footer.span("total"),
        ]
    )
    return tuple(spans)


def mask_form_numeric_locators(text: str) -> str:
    """Blank locator digits without deleting or moving any source character."""
    chars = list(text)
    for start, end in form_numeric_locator_spans(text):
        chars[start:end] = " " * (end - start)
    return "".join(chars)
