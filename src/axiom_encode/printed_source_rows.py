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
class SamePageFloor:
    """Printed operand order and raw spans only; never execution evidence."""

    target: int
    left: int
    right: int
    operation_span: tuple[int, int]
    operand_spans: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class AnnualFormYearRole:
    """Literal reporting-year coordinates; neither arithmetic nor source credit."""

    owner_span: tuple[int, int]
    year_span: tuple[int, int]
    header_span: tuple[int, int]


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
    nonoperative_header_spans: tuple[tuple[int, int], ...] = ()
    same_page_floors: tuple[SamePageFloor, ...] = ()
    note_marker_spans: tuple[tuple[int, int], ...] = ()
    annual_form_year_roles: tuple[AnnualFormYearRole, ...] = ()


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
    # Keep physical record spans separate: detached cells must not absorb the
    # intervening operative text. This describes source syntax, not a proof.
    local_roles = {
        6: ("row6_cell", "row6_add"),
        7: ("row7",),
        11: ("row11_add",),
        12: ("row12",),
    }
    floors = tuple(
        SamePageFloor(
            target=target,
            left=left,
            right=right,
            operation_span=records[f"row{target}"][:2],
            operand_spans=tuple(
                tuple(records[role][:2] for role in local_roles[row])
                for row in (left, right)
            ),
        )
        for target, left, right in ((12, 7, 11), (13, 6, 12))
    )
    return PrintedRowStructure(
        boundaries=tuple(sorted(boundaries)),
        atomic_spans=atomic,
        heading_dash=None,
        additions=additions,
        typography_spans=(),
        ownership_spans=ownership,
        numeric_locator_spans=tuple(sorted(locators)),
        nonoperative_header_spans=(records["header"][:2],),
        same_page_floors=floors,
    )


# Complete four-page transcription. Literal records describe source layout only.
_MULTIPAGE_T936_RECORDS = (
    (
        "header",
        "CRA Form T936 2025 Calculation of Cumulative Net Investment Loss\n\nProtected B when completed\nCalculation of Cumulative Net Investment Loss (CNIL) to December 31, 2025\n",
    ),
    (
        "scope",
        "Use this form if you had any investment income or investment expenses for 2025.\n",
    ),
    (
        "cnil_effect",
        "Your calculation of cumulative net investment loss (CNIL) reduces the amount of your cumulative gains limit for the year and\nmay affect the allowable amount of your capital gains deduction.\n",
    ),
    (
        "concessive_duty",
        "Even if you are not claiming a capital gains deduction in 2025, you should still complete this form if you had any investment\nincome or expenses in 2025.\n",
    ),
    (
        "records_duty",
        "The balance in your CNIL account represents a cumulative total, so you may need this information in a future year. Keep a\ncopy for your records and attach another copy to your return.\n",
    ),
    (
        "chart_referral",
        "Note: If, in 2025, you have capital gains other than from the disposition of qualified farm or fishing property or qualified small\nbusiness corporation shares, first complete Chart A on page 4 of this form to determine if you have additional investment\nincome to include when you calculate your CNIL.\n\n\n",
    ),
    ("part1_heading1", "Part 1 – Investment expenses claimed on your 2025 return\n\n"),
    ("part1_heading2", "Part 1 – Investment expenses claimed on your 2025 return\n"),
    ("part1_row1", "Carrying charges and interest expenses (line 22100) 1\n"),
    ("part1_row2", "Net rental losses (line 12600) + 2\n"),
    (
        "part1_row3",
        "Limited or non-active partnership losses (line 12200) other than allowable capital losses + 3\n",
    ),
    (
        "part1_row4",
        "Limited partnership losses of other years after 1985 (line 25100) + 4\n",
    ),
    ("part1_row5", "50% of exploration and development expenses (line 22400) + 5\n"),
    (
        "part1_row6",
        'Any other investment expenses claimed in 2025 to earn property income\n(see the list "Other investment expenses" on the next page) 68080 + 6\n',
    ),
    (
        "part1_row7",
        'Additional investment expenses: if you did not complete Chart A on page 4 of this form, enter "0".\nOtherwise, enter whichever is less: line 20 in Chart A or line 25300 of your return. + 7\n',
    ),
    (
        "part1_totalA",
        "Add lines 1 to 7. Total investment expenses claimed in 2025 = A\n\n\n",
    ),
    ("part2_heading1", "Part 2 – Investment income reported on your 2025 return\n\n"),
    ("part2_heading2", "Part 2 – Investment income reported on your 2025 return\n"),
    ("part2_row8", "Investment income (lines 12000 and 12100) 8\n"),
    (
        "part2_row9",
        "Net rental income, including recaptured capital cost allowance (line 12600) + 9\n",
    ),
    (
        "part2_row10",
        "Net income from limited or non-active partnership (line 12200) other than taxable capital gains + 10\n",
    ),
    (
        "part2_row11",
        'Any other property income reported in 2025 (see the list "Other property income" on the next page)\nincluding annuity payments taxable under paragraph 56(1)(d) minus the capital portion deducted\nunder paragraph 60(a) 68100 + 11\n',
    ),
    (
        "part2_row12",
        "50% of income from the recovery of exploration and development expenses (line 13000) 68110 + 12\n",
    ),
    (
        "part2_row13",
        'Additional investment income: if you did not complete Chart A on page 4 of this form, enter "0".\nOtherwise, enter the amount from line 20 in Chart A. + 13\n',
    ),
    (
        "part2_totalB",
        "Add lines 8 to 13. Total investment income reported in 2025 = B\n",
    ),
    ("reserved_area", "Do not use this area\n68130\n"),
    (
        "page1_footer",
        "T936 E (25) (Ce formulaire est disponible en français.) Page 1 of 4\n\n",
    ),
    ("expenses_include", "Other investment expenses\nInclude:\n"),
    ("expense_inducement", "(cid:129) repayments of inducements\n"),
    ("expense_refund", "(cid:129) repayments of refund interest\n"),
    (
        "expense_agreement",
        "(cid:129) sale of agreement for sale, or mortgage or hypothecary claim included in proceeds of disposition in a previous year under\nsubsection 20(5)\n",
    ),
    (
        "expense_foreign_tax",
        "(cid:129) foreign non-business tax under subsections 20(11) and 20(12)\n",
    ),
    (
        "expense_insurance",
        "(cid:129) life insurance premiums deducted from property income\n",
    ),
    (
        "expense_films",
        "(cid:129) capital cost allowance claimed on certified films and videotapes\n",
    ),
    (
        "expense_partner",
        "(cid:129) farming or fishing losses claimed by a non-active partner or a limited partner\n",
    ),
    (
        "expense_uncollectible",
        "(cid:129) the uncollectible portion of proceeds from dispositions of depreciable property (except timber resources, passenger vehicles\nthat cost more than $30,000 for vehicles acquired prior to 2022, $34,000 for vehicles acquired in 2022, $36,000 for vehicles\nacquired in 2023, $37,000 for vehicles acquired in 2024, and $38,000 for vehicles acquired in 2025, and zero-emission\npassenger vehicles (ZEPV) that cost more than $55,000 for ZEPV acquired after March 18, 2019, $59,000 for ZEPV acquired\nin 2022, and $61,000 for ZEPV acquired after 2022)\n",
    ),
    ("expenses_exclude", "Do not include:\n"),
    (
        "excluded_business_expense",
        "(cid:129) expenses incurred to earn business income\n",
    ),
    (
        "excluded_shareholder_expense",
        "(cid:129) repayment of shareholders' loans deducted under paragraph 20(1)(j)\n",
    ),
    (
        "excluded_interest_group",
        "(cid:129) interest paid on money borrowed to:\n– buy an income averaging annuity contract\n– pay a premium under a registered retirement savings plan\n– make a contribution to a registered pension plan\n– make a contribution to a deferred profit-sharing plan\n",
    ),
    ("income_include", "Other property income\nInclude:\n"),
    (
        "income_insurance",
        "(cid:129) amounts from insurance proceeds for the recapture of capital cost allowance\n(other than amounts already included on line 9)\n",
    ),
    (
        "income_grants",
        "(cid:129) home insulation or energy conversion grants under paragraph 12(1)(u)\n",
    ),
    (
        "income_inducement",
        "(cid:129) payments received as an inducement or reimbursement\n",
    ),
    (
        "income_shareholder",
        "(cid:129) income from the appropriation of property to a shareholder\n",
    ),
    (
        "income_partner",
        "(cid:129) farming and fishing income reported by a non-active or a limited partner\n",
    ),
    ("income_trust", "(cid:129) other income from a trust\n"),
    (
        "income_prior_partnership_losses",
        "(cid:129) allowable capital losses included in partnership losses of other years after 1985\n",
    ),
    ("income_agriinvest", "(cid:129) amounts withdrawn from AgriInvest Fund 2\n"),
    (
        "income_death_benefits",
        "(cid:129) Canada Pension Plan (CPP) or Quebec Pension Plan (QPP) death benefit payments reported on your\nincome tax and benefit return\n",
    ),
    ("income_exclude", "Do not include:\n"),
    (
        "excluded_business_income",
        "(cid:129) income amounts that relate to business income\n",
    ),
    (
        "excluded_averaging_annuity",
        "(cid:129) payments received from an income averaging annuity contract\n",
    ),
    (
        "excluded_deferred_annuity",
        "(cid:129) payments received from an annuity contract bought under a deferred profit-sharing plan\n",
    ),
    (
        "excluded_shareholder_income",
        "(cid:129) shareholders' loans included in income under subsection 15(2)\n",
    ),
    ("page2_footer", "T936 E (25) Page 2 of 4\n\n"),
    ("page3_protection", "Protected B when completed\n\n\n"),
    ("part3_heading1", "Part 3 – Cumulative net investment loss (CNIL)\n\n"),
    ("part3_heading2", "Part 3 – Cumulative net investment loss (CNIL)\n"),
    (
        "part3_row14",
        "Total investment expenses claimed in 2025 (line A of Part 1) 14\n",
    ),
    (
        "part3_row15",
        "Total investment expenses claimed in previous years after 1987:\nAmount from line 16 in Part 3 of Form T936 for 2024 (if you did not\ncomplete Form T936 for 2024, see note (1)) + 15\n",
    ),
    ("part3_row16", "Line 14 plus line 15 = 16\n"),
    ("part3_row17", "Amount from line 17218 of Form T2048 for 2024 to 2025 – 17\n"),
    (
        "part3_row18",
        "Cumulative investment expenses (line 16 minus line 17) = (cid:379) 18\n",
    ),
    ("part3_row19", "Total investment income reported in 2025 (line B of Part 2) 19\n"),
    (
        "part3_row20",
        "Total investment income reported in previous years after 1987:\nAmount from line 19 of Part 3 of Form T936 for 2024. (if you did not\ncomplete Form T936 for 2024, see note (2)) + 20\n",
    ),
    (
        "part3_row21",
        "Cumulative investment income (total of lines 19 and 20) = (cid:379) – 21\n",
    ),
    (
        "part3_finalC",
        'Line 18 minus line 21 (if negative, enter "0") CNIL to December 31, 2025 = C\n',
    ),
    (
        "reporting_duty",
        "If you are claiming a capital gains deduction on your 2025 return, report the amount from line C on line 16 of Step 2\nof Form T657.\n",
    ),
    (
        "note1",
        "(1) To calculate your total investment expenses from previous years, complete Part 1 of Form T936 for each year from\n1988 to 2024 in which you had investment expenses (do not complete line 7 for 1988 to 1991). Add the amounts from\nline A and enter the total on line 15 above.\n",
    ),
    (
        "note2",
        "(2) To calculate your total investment income from previous years, complete Part 2 of Form T936 for each year from\n1988 to 2024 in which you had investment income (do not complete line 13 for 1988 to 1991). Add the amounts from\nline B and enter the total on line 20 above.\n",
    ),
    ("page3_footer", "T936 E (25) Page 3 of 4\n\n"),
    ("page4_protection", "Protected B when completed\n"),
    ("chart_heading", "Chart A\n"),
    (
        "chart_row1",
        'Enter the amount from line 19900 of Schedule 3, Capital Gains or Losses (if negative, show the\namount in brackets). If the amount is "0", do not complete lines 2 to 17, and enter "0" on line 20. 1\n',
    ),
    ("chart_row2", "Amount from line 17215 of Form T2048 – 2\n"),
    ("chart_row3", "Line 1 minus line 2 = 3\n"),
    ("chart_row4", "Amount from line 10700 of Schedule 3 4\n"),
    ("chart_row5", "Amount from line 11000 of Schedule 3 + 5\n"),
    ("chart_row6", "Amount from line 12400 of Schedule 3 + 6\n"),
    ("chart_row7", "Add lines 4 to 6 (if negative, show in brackets). = 7\n"),
    (
        "chart_row8",
        "If you reported an amount on line 19200 of Schedule 3, enter the amount\nfrom line 13 of Form T2017, Summary of Reserves on Dispositions of\nCapital Property (if negative, show in brackets). + 8\n",
    ),
    ("chart_row9", "Line 7 plus line 8 (if negative, show in brackets) = 9\n"),
    ("chart_row10", "Inclusion rate × 50% 10\n"),
    (
        "chart_row11",
        'Line 9 multiplied by the percentage on line 10\n(if negative, enter "0") = 11\n',
    ),
    ("chart_row12", "Amount from line 17216 of Form T2048 – 12\n"),
    ("chart_row13", "Line 11 minus line 12 = (cid:379) – 13\n"),
    ("chart_row14", 'Line 3 minus line 13 (if negative, enter "0") = 14\n'),
    (
        "chart_zero14",
        'If the total from line 14 is "0", do not complete lines 15 to 19, and enter "0" on line 20.\n',
    ),
    (
        "chart_t3_gate",
        'If the total from line 14 includes an amount from a T3 slip, complete lines 15 to 19 below.\nOtherwise, enter "0" on line 19 and continue at line 20.\n',
    ),
    ("chart_row15", "Enter the amount from box 21 of all 2025 T3 slips. 68140 15\n"),
    ("chart_row16", "Enter the amount from box 30 of all 2025 T3 slips. – 16\n"),
    ("chart_row17", "Line 15 minus line 16 68150 = 17\n"),
    ("chart_row18", "Inclusion rate × 50% 18\n"),
    (
        "chart_row19",
        "Line 17 multiplied by the percentage on line 18 = (cid:379) – 19\n",
    ),
    (
        "chart_row20",
        'Line 14 minus line 19 (if negative, enter "0") Additional investment income = 20\n',
    ),
    ("privacy_duty", "See the privacy notice on your return.\n"),
    ("page4_footer", "T936 E (25) Page 4 of 4"),
)


_MULTIPAGE_T936_LOCATORS = (
    ("part1_row2", "es (line 12600) + 2\n", 18, "2"),
    ("part1_row3", " capital losses + 3\n", 18, "3"),
    ("part1_row4", "85 (line 25100) + 4\n", 18, "4"),
    ("part1_row5", "es (line 22400) + 5\n", 18, "5"),
    ("part1_row6", "ext page) 68080 + 6\n", 18, "6"),
    ("part1_row7", "of your return. + 7\n", 18, "7"),
    ("part2_row9", "ce (line 12600) + 9\n", 18, "9"),
    ("part2_row10", "e capital gains + 10\n", 18, "10"),
    ("part2_row11", "aph 60(a) 68100 + 11\n", 18, "11"),
    ("part2_row12", "ne 13000) 68110 + 12\n", 18, "12"),
    ("part2_row13", " 20 in Chart A. + 13\n", 18, "13"),
    ("part3_row15", ", see note (1)) + 15\n", 18, "15"),
    ("part3_row16", "14 plus line 15 = 16\n", 18, "16"),
    ("part3_row20", ", see note (2)) + 20\n", 18, "20"),
    ("chart_row3", " 1 minus line 2 = 3\n", 18, "3"),
    ("chart_row5", "0 of Schedule 3 + 5\n", 18, "5"),
    ("chart_row6", "0 of Schedule 3 + 6\n", 18, "6"),
    ("chart_row7", "w in brackets). = 7\n", 18, "7"),
    ("chart_row8", "w in brackets). + 8\n", 18, "8"),
    ("chart_row9", "ow in brackets) = 9\n", 18, "9"),
    ("chart_row11", 'ive, enter "0") = 11\n', 18, "11"),
    ("chart_row14", 'ive, enter "0") = 14\n', 18, "14"),
    ("chart_row17", "s line 16 68150 = 17\n", 18, "17"),
    ("chart_row20", "vestment income = 20\n", 18, "20"),
    ("chart_referral", "e Chart A on page 4 of this form to d", 18, "4"),
    ("part1_row1", "nses (line 22100) 1\n", 18, "1"),
    ("part1_row6", "on the next page) 68080 + 6\n", 18, "68080"),
    ("part1_row7", "e Chart A on page 4 of this form, ent", 18, "4"),
    ("part2_row8", " 12000 and 12100) 8\n", 18, "8"),
    ("part2_row11", "r paragraph 60(a) 68100 + 11\n", 18, "68100"),
    ("part2_row12", "nses (line 13000) 68110 + 12\n", 18, "68110"),
    ("part2_row13", "e Chart A on page 4 of this form, ent", 18, "4"),
    ("reserved_area", "not use this area\n68130\n", 18, "68130"),
    ("income_agriinvest", "m AgriInvest Fund 2\n", 18, "2"),
    ("part3_row14", "5 (line A of Part 1) 14\n", 18, "1"),
    ("part3_row14", "line A of Part 1) 14\n", 18, "14"),
    ("part3_row17", "or 2024 to 2025 – 17\n", 18, "17"),
    ("part3_row18", "e 17) = (cid:379) 18\n", 18, "18"),
    ("part3_row19", "5 (line B of Part 2) 19\n", 18, "2"),
    ("part3_row19", "line B of Part 2) 19\n", 18, "19"),
    ("part3_row21", "20) = (cid:379) – 21\n", 18, "21"),
    ("chart_row1", 'r "0" on line 20. 1\n', 18, "1"),
    ("chart_row2", "5 of Form T2048 – 2\n", 18, "2"),
    ("chart_row4", "700 of Schedule 3 4\n", 18, "4"),
    ("chart_row10", "lusion rate × 50% 10\n", 18, "10"),
    ("chart_row12", "6 of Form T2048 – 12\n", 18, "12"),
    ("chart_row13", " 12 = (cid:379) – 13\n", 18, "13"),
    ("chart_row15", "e amount from box 21 of all 2025 T3 sl", 18, "21"),
    ("chart_row15", "ll 2025 T3 slips. 68140 15\n", 18, "68140"),
    ("chart_row15", "5 T3 slips. 68140 15\n", 18, "15"),
    ("chart_row16", "e amount from box 30 of all 2025 T3 sl", 18, "30"),
    ("chart_row16", " 2025 T3 slips. – 16\n", 18, "16"),
    ("chart_row18", "lusion rate × 50% 18\n", 18, "18"),
    ("chart_row19", " 18 = (cid:379) – 19\n", 18, "19"),
)


def _multipage_t936_rows(text: str) -> PrintedRowStructure | None:
    records: dict[str, tuple[int, int]] = {}
    cursor = 0
    for role, literal in _MULTIPAGE_T936_RECORDS:
        if not text.startswith(literal, cursor):
            return None
        records[role] = (cursor, cursor + len(literal.rstrip("\r\n")))
        cursor += len(literal)
    if cursor != len(text):
        return None
    # Include/exclude headings govern every item in their block. Internal
    # literal records corroborate offsets, not independent source ownership.
    category_groups = tuple(
        (records[first][0], records[last][1])
        for first, last in (
            ("expenses_include", "expense_uncollectible"),
            ("expenses_exclude", "excluded_interest_group"),
            ("income_include", "income_death_benefits"),
            ("income_exclude", "excluded_shareholder_income"),
        )
    )
    header_roles = (
        "header",
        "part1_heading1",
        "part1_heading2",
        "part2_heading1",
        "part2_heading2",
        "reserved_area",
        "page1_footer",
        "page2_footer",
        "page3_protection",
        "part3_heading1",
        "part3_heading2",
        "page3_footer",
        "page4_protection",
        "chart_heading",
        "page4_footer",
    )
    atomic_roles = (
        "part1_row7",
        "part2_row13",
        "part3_row15",
        "part3_row20",
        "part3_finalC",
        "note1",
        "note2",
        "expense_uncollectible",
        "excluded_interest_group",
        "chart_row1",
        "chart_row7",
        "chart_row8",
        "chart_row9",
        "chart_row11",
        "chart_row14",
        "chart_zero14",
        "chart_t3_gate",
        "chart_row20",
    )
    atomic = (
        *category_groups,
        records["part1_totalA"],
        records["part2_totalB"],
        *(records[role] for role in atomic_roles),
    )
    ownership = (
        tuple(
            span
            for role, span in records.items()
            if role not in header_roles
            and not any(a <= span[0] and span[1] <= b for a, b in category_groups)
        )
        + category_groups
    )
    boundaries = {0, len(text)}
    boundaries.update(
        point
        for span in records.values()
        for point in span
        if not any(a < point < b for a, b in atomic)
    )
    # A title separator is not subtraction. Only these six literal heading
    # glyphs are typography; every operative minus and multiplication remains.
    heading_dashes = []
    for role in (
        "part1_heading1",
        "part1_heading2",
        "part2_heading1",
        "part2_heading2",
        "part3_heading1",
        "part3_heading2",
    ):
        start, end = records[role]
        literal = text[start:end]
        if literal.count("–") != 1:
            return None
        pos = start + literal.index("–")
        heading_dashes.append((pos, pos + 1))
    # A caption glyph identifies the operand column; the linked application
    # row carries the actual ordered computation. The full records above must
    # match, including its operands, rate, and any floor or conditional guard.
    caption_glyphs = []
    for role, glyph, applied, expected in (
        (
            "part1_row2",
            "+",
            "part1_totalA",
            "Add lines 1 to 7. Total investment expenses claimed in 2025 = A",
        ),
        (
            "part1_row3",
            "+",
            "part1_totalA",
            "Add lines 1 to 7. Total investment expenses claimed in 2025 = A",
        ),
        (
            "part1_row4",
            "+",
            "part1_totalA",
            "Add lines 1 to 7. Total investment expenses claimed in 2025 = A",
        ),
        (
            "part1_row5",
            "+",
            "part1_totalA",
            "Add lines 1 to 7. Total investment expenses claimed in 2025 = A",
        ),
        (
            "part1_row6",
            "+",
            "part1_totalA",
            "Add lines 1 to 7. Total investment expenses claimed in 2025 = A",
        ),
        (
            "part1_row7",
            "+",
            "part1_totalA",
            "Add lines 1 to 7. Total investment expenses claimed in 2025 = A",
        ),
        (
            "part2_row9",
            "+",
            "part2_totalB",
            "Add lines 8 to 13. Total investment income reported in 2025 = B",
        ),
        (
            "part2_row10",
            "+",
            "part2_totalB",
            "Add lines 8 to 13. Total investment income reported in 2025 = B",
        ),
        (
            "part2_row11",
            "+",
            "part2_totalB",
            "Add lines 8 to 13. Total investment income reported in 2025 = B",
        ),
        (
            "part2_row12",
            "+",
            "part2_totalB",
            "Add lines 8 to 13. Total investment income reported in 2025 = B",
        ),
        (
            "part2_row13",
            "+",
            "part2_totalB",
            "Add lines 8 to 13. Total investment income reported in 2025 = B",
        ),
        (
            "chart_row5",
            "+",
            "chart_row7",
            "Add lines 4 to 6 (if negative, show in brackets). = 7",
        ),
        (
            "chart_row6",
            "+",
            "chart_row7",
            "Add lines 4 to 6 (if negative, show in brackets). = 7",
        ),
        (
            "part3_row17",
            "–",
            "part3_row18",
            "Cumulative investment expenses (line 16 minus line 17) = (cid:379) 18",
        ),
        ("chart_row2", "–", "chart_row3", "Line 1 minus line 2 = 3"),
        ("chart_row12", "–", "chart_row13", "Line 11 minus line 12 = (cid:379) – 13"),
        (
            "chart_row10",
            "×",
            "chart_row11",
            'Line 9 multiplied by the percentage on line 10\n(if negative, enter "0") = 11',
        ),
        (
            "chart_row18",
            "×",
            "chart_row19",
            "Line 17 multiplied by the percentage on line 18 = (cid:379) – 19",
        ),
    ):
        a, b = records[applied]
        if text[a:b] != expected:
            return None
        start, end = records[role]
        caption = text[start:end]
        if caption.count(glyph) != 1:
            return None
        pos = start + caption.index(glyph)
        caption_glyphs.append((pos, pos + 1))
    locators = []
    record_literals = dict(_MULTIPAGE_T936_RECORDS)
    for role, context, offset, token in _MULTIPAGE_T936_LOCATORS:
        start, end = records[role]
        literal = record_literals[role]
        if (
            literal.count(context) != 1
            or context[offset : offset + len(token)] != token
        ):
            return None
        pos = start + literal.index(context) + offset
        if not start <= pos < pos + len(token) <= end:
            return None
        locators.append((pos, pos + len(token)))
    if len(locators) != len(set(locators)):
        return None
    annual_roles = []
    for role in (
        "part1_row6",
        "part1_totalA",
        "part2_row11",
        "part2_totalB",
        "part3_row14",
        "part3_row19",
    ):
        start, end = records[role]
        match = re.search(r"(?:claimed|reported) in (?P<year>2025)\b", text[start:end])
        if match is None:
            return None
        annual_roles.append(
            AnnualFormYearRole(
                (start, end),
                (start + match.start("year"), start + match.end("year")),
                (records["header"][0], text.index("\n")),
            )
        )
    return PrintedRowStructure(
        boundaries=tuple(sorted(boundaries)),
        atomic_spans=tuple(sorted(set(atomic))),
        heading_dash=None,
        additions=tuple(
            records[role] for role in ("part1_totalA", "part2_totalB", "chart_row7")
        ),
        typography_spans=tuple(sorted((*heading_dashes, *caption_glyphs))),
        ownership_spans=tuple(sorted(ownership)),
        nonoperative_header_spans=tuple(records[role] for role in header_roles),
        numeric_locator_spans=tuple(sorted(locators)),
        annual_form_year_roles=tuple(annual_roles),
        note_marker_spans=tuple(
            (records[role][0], records[role][0] + 3) for role in ("note1", "note2")
        ),
    )


def recognize_printed_rows(
    text: str, *, allow_detached: bool = False, allow_multipage: bool = False
) -> PrintedRowStructure | None:
    if allow_multipage is True:
        multipage = _multipage_t936_rows(text)
        if multipage is not None:
            return multipage
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
