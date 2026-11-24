"""Closed Schedule N physical source records; no formula or coverage authority.

Preserves the resolver's non-reading-order extraction and every separator.
An exact recognized form is only an input to subsequent semantic binding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-n-investment-expense-adjustment"
_RECORD_TEXT: tuple[tuple[str, str], ...] = (
    (
        "resolver_title",
        "Revenu Quebec Schedule N 2025 - Adjustment of Investment Expenses",
    ),
    ("physical_revision", "T P -1. D. N-V (2025-12)\nT"),
    ("part_a_heading", "Adjustment of Investment Expenses\nA"),
    ("a_row10", "10"),
    ("a_addends", "+ 12\n+ 14\n+ 16\n="),
    ("a_total18", "18"),
    ("income_heading", "Investment income"),
    (
        "income_captions_operations",
        "Dividends from taxable Canadian corporations (line 128 of your return)\nInterest and other investment income (line 130 of your return)\nIncome from a partnership of which you were a specified member\n(included on line 29 of Schedule L or on line 136 of your return)\nIncome accrued under a life insurance policy, RL-3 slip, box J\nRecovery of resource deductions (see line 154, point 11, in the guide)\nmultiplied by 50%\nOrdinary annuities, RL-2 slip, box B,\nif code “RO” appears in the “Provenance des revenus” box\nOther property income allocated by a trust and property income allocated\nto shareholders\nTaxable capital gains (see line 260 in the guide)\nAdd lines 20 through 34.\nSubtract line 36 from line 18. If the result is negative, enter 0.\nCarry the result to line 260 of your return.\x08",
    ),
    ("part_c_marker", "C"),
    ("schedule_marker", "N"),
    ("expense_heading", "Investment expenses"),
    (
        "expense_captions_operations",
        "Loss from a partnership of which you were a specified member\n(included on line 29 of Schedule L or on line 136 of your return)\nCarrying charges and interest expenses (line 231 of your return)\nDeduction for exploration and development expenses (see line 260 in the guide)\nOther expenses incurred to earn property income\n(see line 260 in the guide)\nAdd lines 10 through 16.",
    ),
    ("part_b_marker", "B"),
    ("schedule_label", "SCHEDULE"),
    ("b_row20", "20"),
    ("b_addends", "+ 22\n+ 24\n+ 26\n+ 28\n+ 30\n+ 32\n+ 34\n="),
    ("b_total36", "36"),
    ("b_adjustment40", "Adjustment of investment expenses = 40"),
    ("other_expenses_heading", "Other investment expenses"),
    (
        "part_c_captions_operations",
        "Complete this part only if you entered an amount on lines 289 and 290 of your return.\nLimited partnership loss\n(included on line 289 of your return)\nNet capital losses from other years (see line 276, point 9, in the guide)\nAdd lines 50 and 52.\nAmount from line 36 above\nAmount from line 18 above\nSubtract line 58 from line 56. If the result is negative, enter 0.\nSubtract line 60 from line 54. If the result is negative, enter 0.\nCarry the result to line 276 of your return.\x08",
    ),
    ("part_d_marker", "D"),
    ("c_row50", "50"),
    ("c_addend52", "+ 52\n="),
    ("c_total54", "54"),
    ("c_copy56", "56"),
    ("c_subtraction60", "– 58\n60"),
    ("c_adjustment64", "Adjustment of other investment expenses = 64"),
    ("carryforward_heading", "Amount that can be carried forward"),
    (
        "part_d_captions_operations",
        "Unused portion of the adjustment of investment expenses (see line 252 in the guide)\nAmount from line 40 above\nAmount from line 64 above\nAdd lines 70 through 74.\nAmount from line 252 of your 2025 return\nSubtract line 78 from line 76.\n\x08",
    ),
    (
        "document_duties",
        "Enclose this schedule with your return.\nIf you want to keep a copy for your records, see the guide.",
    ),
    ("d_row70", "70"),
    ("d_operators", "+\n+\n=\n–"),
    ("d_remaining_rows", "72\n74\n76\n78"),
    ("d_total80", "Amount that can be carried forward = 80"),
    ("printer_identifier", "Y4N1 ZZ 89527849"),
    ("footer_schedule_marker", "N"),
)


@dataclass(frozen=True)
class InvestmentAdjustmentRecord:
    role: str
    span: tuple[int, int]


@dataclass(frozen=True)
class InvestmentAdjustmentSource:
    source: str
    citation: str
    source_sha256: str
    records: tuple[InvestmentAdjustmentRecord, ...]
    separators: tuple[tuple[int, int], ...]

    def bound(self, source: str, citation: str) -> bool:
        return recognize_investment_adjustment_source(source, citation) == self


def recognize_investment_adjustment_source(
    source: str, citation: str = CITATION
) -> InvestmentAdjustmentSource | None:
    """Recognize the entire literal physical form without normalizing any bytes."""
    if type(source) is not str or citation != CITATION:
        return None
    offset = 0
    records = []
    separators = []
    for index, (role, text) in enumerate(_RECORD_TEXT):
        if not source.startswith(text, offset):
            return None
        end = offset + len(text)
        records.append(InvestmentAdjustmentRecord(role, (offset, end)))
        offset = end
        if index < len(_RECORD_TEXT) - 1:
            if source[offset : offset + 2] != "\n\n":
                return None
            separators.append((offset, offset + 2))
            offset += 2
    if offset != len(source):
        return None
    return InvestmentAdjustmentSource(
        source,
        citation,
        hashlib.sha256(source.encode()).hexdigest(),
        tuple(records),
        tuple(separators),
    )
