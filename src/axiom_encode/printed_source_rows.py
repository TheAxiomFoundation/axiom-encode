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

# Closed terminal referrals in validated printed rows; never arbitrary slash
# or dash expressions. These spans affect formula classification only.
_URL_REFERRAL = re.compile(
    r"\(go to (?P<url>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}/[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*)\)"
    r"(?: \^| Line [1-9]\d{4}: Total \^)?$"
)
# A separate closed operative row: preserve all arithmetic and advisory text.
_ARITHMETIC_URL_REFERRAL = re.compile(
    r"Line (?P<left>[1-9]\d{0,2}) minus line (?P<right>[1-9]\d{0,2}) "
    r'\(if negative, enter "0"\) If negative, you may have a non-capital loss '
    r"\(see Form [A-Z][A-Z0-9]{1,9}\) and the negative amount is to be used "
    r"for certain calculations \(go to "
    r"(?P<url>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}/[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*)\) "
    r"[A-Z][A-Za-z ]* = \^"
)

_OTHER_DESCRIPTOR = re.compile(
    r"[A-Z][A-Za-z ]+ (?P<dash>-) other "
    r"\(see the [A-Z]{1,4}\d{1,4}[A-Z]{0,4} slip\) \^"
)


@dataclass(frozen=True)
class PrintedRowStructure:
    boundaries: tuple[int, ...]
    atomic_spans: tuple[tuple[int, int], ...]
    heading_dash: tuple[int, int] | None
    additions: tuple[tuple[int, int], ...]
    typography_spans: tuple[tuple[int, int], ...]
    ownership_spans: tuple[tuple[int, int], ...]
    arithmetic_referral_spans: tuple[tuple[int, int], ...] = ()
    numeric_locator_spans: tuple[tuple[int, int], ...] = ()


# Exact detached Schedule 2 transcription. Every record includes its literal
# separators; no operative prose or unknown suffix may be skipped. Physical
# order differs from calculation order (notably rows 3, 6, 7, 10 and 11).
_DETACHED_SCHEDULE2_RECORDS = (
    (
        "header",
        "CRA Schedule 2 2025 Federal Amounts Transferred from your Spouse or Common-Law Partner\n\nT1-2025\n\nFederal Amounts Transferred from your\nSpouse or Common-Law Partner\n\nSchedule 2\nProtected B when completed\n\n",
    ),
    (
        "scope",
        "Complete this schedule to claim a transfer of the unused part of your spouse's or common-law partner's federal amounts below.\nYou may be able to claim all or part of certain amounts that your spouse or common-law partner qualifies for if they do not need\nto use them to reduce their federal tax to zero.\nComplete the provincial or territorial schedule (S2) to calculate your provincial or territorial amounts.\n",
    ),
    (
        "separation",
        "If you were separated because of a breakdown in your relationship for a period of 90 days or more including December 31, 2025,\nyour spouse or common-law partner cannot transfer any unused amounts to you.\n",
    ),
    (
        "filer",
        "If your spouse or common-law partner is filing a return, use the amounts entered on their return.\n",
    ),
    (
        "nonfiler",
        "If your spouse or common-law partner is not filing a return, use the amounts they would have entered on their return, schedules,\nand worksheets as if they were filing a return, and attach their information slips.\n",
    ),
    ("attach_schedule", "Attach a copy of this schedule to your paper return.\n"),
    (
        "row1",
        "Age amount (if your spouse or common-law partner was born in 1960 or earlier):\nEnter the amount from line 30100 of their return.\n\n35200\n\n1\n\n",
    ),
    (
        "row2",
        "Canada caregiver amount for infirm children under 18 years of age:\nEnter the amount from line 30500 of their return.\n\n35300 +\n\n2\n\n",
    ),
    ("row3_cap_cell", "(maximum $2,000) 35500 +\n\n3\n\n"),
    (
        "row4",
        "Disability amount:\nEnter the amount from line 31600 of their return. (1)\n\n35700 +\n\n4\n\n",
    ),
    (
        "row5",
        "Tuition amount:\nEnter the federal amount designated to you from their forms T2202, TL11A, or TL11C. (2)\n\n36000 +\n\n5\n\n",
    ),
    ("row6_cell", "=\n\n6\n\n"),
    (
        "row12",
        "Spouse's or common-law partner's adjusted taxable income:\nLine 7 minus line 11 (if negative, enter \"0\")\n\n–\n\n12\n\n",
    ),
    (
        "row13",
        'Line 6 minus line 12 (if negative, enter "0")\nEnter this amount on line 32600 of your return.\n\n=\n\n13\n\n',
    ),
    (
        "row3_caption",
        "Pension income amount:\nEnter the amount from line 31400 of their return.\n\n",
    ),
    ("row6_add", "Add lines 1 to 5.\n"),
    (
        "row7",
        "Spouse's or common-law partner's taxable income:\nEnter the amount from line 26000 of their return on line 7 if it is $57,375 or less.\nIf it is more than $57,375, enter the result of the following calculation:\namount from line 76 of their return\n÷ 14.5% =\n",
    ),
    ("row8", "Amount from line 30000 of their return\n8\n"),
    ("row9", "Amount from line 96 of their return\n+\n9\n"),
    ("row10_caption", "Amount from line 32300 of their return\n"),
    ("row11_add", "Add lines 8 to 10.\n\n"),
    ("tail_cells", "+\n=\n\n10\nŹ\n\n7\n\n–\n\n11\n\nŹ\n"),
    (
        "final_field_caption",
        "36100 =\nFederal amounts transferred from your\nspouse or common-law partner\n\n",
    ),
    (
        "footnote1",
        "(1) If this is a new claim for the disability amount, attach a completed and certified Form T2201, Disability Tax Credit\n\nCertificate, to your paper return.\nBefore assessing your return, the Canada Revenue Agency (CRA) will review your claim to see if your spouse or\ncommon-law partner is eligible for the disability tax credit. If they were eligible for 2024 and still meet the requirements\nin 2025, you do not need to send the CRA a new Form T2201 to claim this amount. However, you must send the CRA\na new Form T2201 if the previous period of approval ended before 2025 or if the CRA asks you to.\n",
    ),
    (
        "footnote2",
        "(2) The maximum tuition amount your spouse or common-law partner can transfer to you is $5,000 minus the current year\n\namounts they use, even if there is still an unused part for the year.\nYour spouse or common-law partner cannot transfer to you their tuition amounts carried forward from a previous year.\n",
    ),
    ("privacy", "See the privacy notice on your return.\n\n"),
    (
        "footer",
        "5000-S2 E (25)\n\n(Ce formulaire est disponible en français.)\n\nPage 1 of 1",
    ),
)

_DETACHED_SCHEDULE2_LOCATORS = (
    ("row1", "\n\n1\n", "1"),
    ("row2", "\n\n2\n", "2"),
    ("row3_cap_cell", "\n\n3\n", "3"),
    ("row4", "\n\n4\n", "4"),
    ("row5", "\n\n5\n", "5"),
    ("row6_cell", "\n\n6\n", "6"),
    ("tail_cells", "\n\n7\n", "7"),
    ("row8", "\n8\n", "8"),
    ("row9", "\n9\n", "9"),
    ("tail_cells", "\n\n10\n", "10"),
    ("tail_cells", "\n\n11\n", "11"),
    ("row12", "\n\n12\n", "12"),
    ("row13", "\n\n13\n", "13"),
    ("row1", "35200", "35200"),
    ("row2", "35300", "35300"),
    ("row3_cap_cell", "35500", "35500"),
    ("row4", "35700", "35700"),
    ("row5", "36000", "36000"),
    ("final_field_caption", "36100", "36100"),
    ("row6_add", "lines 1", "1"),
    ("row6_add", "to 5", "5"),
    ("row11_add", "lines 8", "8"),
    ("row11_add", "to 10", "10"),
    ("row12", "Line 7", "7"),
    ("row12", "line 11", "11"),
    ("row13", "Line 6", "6"),
    ("row13", "line 12", "12"),
    ("row7", "on line 7", "7"),
    ("row4", "(1)", "1"),
    ("footnote1", "(1)", "1"),
    ("row5", "(2)", "2"),
    ("footnote2", "(2)", "2"),
)


def _detached_schedule2_rows(text: str) -> PrintedRowStructure | None:
    records: dict[str, tuple[int, int, str]] = {}
    cursor = 0
    for role, literal in _DETACHED_SCHEDULE2_RECORDS:
        if not text.startswith(literal, cursor):
            return None
        # Content endpoints are derived from the matched literal, never from
        # fixture coordinates. The full literal including separators matched.
        records[role] = (cursor, cursor + len(literal.rstrip()), literal)
        cursor += len(literal)
    if cursor != len(text):
        return None
    table_roles = (
        "row1",
        "row2",
        "row3_cap_cell",
        "row4",
        "row5",
        "row6_cell",
        "row12",
        "row13",
        "row3_caption",
        "row6_add",
        "row7",
        "row8",
        "row9",
        "row10_caption",
        "row11_add",
        "tail_cells",
        "final_field_caption",
    )
    ownership = tuple(records[role][:2] for role in table_roles)
    boundaries = {0, records["header"][1]}
    boundaries.update(point for pair in ownership for point in pair)
    # These within-row conditions remain atomic. Disjoint caption/cell pairs
    # are never joined across another row's operative text.
    atomic = tuple(records[role][:2] for role in ("row1", "row7", "row12", "row13"))
    additions = tuple(records[role][:2] for role in ("row6_add", "row11_add"))
    locators = []
    for role, context, token in _DETACHED_SCHEDULE2_LOCATORS:
        start, _end, literal = records[role]
        if literal.count(context) != 1 or context.count(token) != 1:
            return None
        position = start + literal.index(context) + context.index(token)
        locators.append((position, position + len(token)))
    if len(locators) != len(set(locators)):
        return None
    return PrintedRowStructure(
        boundaries=tuple(sorted(boundaries)),
        atomic_spans=atomic,
        heading_dash=None,
        additions=additions,
        typography_spans=(),
        ownership_spans=ownership,
        numeric_locator_spans=tuple(sorted(locators)),
    )


def recognize_printed_rows(
    text: str, *, allow_detached: bool = False
) -> PrintedRowStructure | None:
    if allow_detached is True:
        detached = _detached_schedule2_rows(text)
        if detached is not None:
            return detached
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
    typography = []
    arithmetic_referrals = []
    for row in rows:
        caption = row["caption"]
        arithmetic_referral = _ARITHMETIC_URL_REFERRAL.fullmatch(caption)
        if arithmetic_referral and row["field"]:
            left, right = (int(arithmetic_referral[k]) for k in ("left", "right"))
            if left in display and right in display and left < right < int(row["row"]):
                a = row.start("caption") + arithmetic_referral.start("url")
                b = row.start("caption") + arithmetic_referral.end("url")
                if not any(x < b and a < y for x, y in quoted):
                    arithmetic_referrals.append((a, b))
        referral = _URL_REFERRAL.search(caption)
        if (
            referral
            and "(" not in caption[: referral.start()]
            and ")" not in caption[: referral.start()]
            and not re.search(
                r"[+*/=−–-]|\b(?:add|subtract|multiply|divide|plus|minus|sum|difference|product|ratio)\b",
                caption[: referral.start()],
                re.I,
            )
        ):
            a = row.start("caption") + referral.start()
            b = row.start("caption") + referral.end()
            if not any(x < b and a < y for x, y in quoted):
                typography.extend(
                    (
                        row.start("caption") + referral.start("url") + m.start(),
                        row.start("caption") + referral.start("url") + m.end(),
                    )
                    for m in re.finditer(r"[-/]", referral["url"])
                )
        descriptor = _OTHER_DESCRIPTOR.fullmatch(caption)
        if (
            descriptor
            and row["field"]
            and not any(x < row.end() and row.start("caption") < y for x, y in quoted)
        ):
            typography.append(
                (
                    row.start("caption") + descriptor.start("dash"),
                    row.start("caption") + descriptor.end("dash"),
                )
            )
    return PrintedRowStructure(
        tuple(sorted(boundaries)),
        tuple(atomic),
        title.span("dash"),
        tuple(additions),
        tuple(typography),
        tuple(
            sorted(
                {
                    next(
                        (
                            span
                            for span in atomic
                            if span[0] <= row.start() and row.end() <= span[1]
                        ),
                        row.span(),
                    )
                    for row in rows
                }
            )
        ),
        tuple(arithmetic_referrals),
    )


def mask_printed_heading_dash(text: str) -> str:
    evidence = recognize_printed_rows(text)
    if evidence is None or evidence.heading_dash is None:
        return text
    a, b = evidence.heading_dash
    return text[:a] + " " * (b - a) + text[b:]
