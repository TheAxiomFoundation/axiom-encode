"""Complete raw retirement-transfer form records, without execution authority.

These records do not authenticate input truth, admit snapshots, credit numeric
values, or establish formula/condition coverage. Every raw character is retained.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

Span = tuple[int, int]

# Closed source shape: gaps are matched literally and remain explicitly unowned.
_PARTS = (
    (
        "header",
        "T P -1. D.Q-V ({year}-12)\nT\n\nSCHEDULE\n\nRetirement Income Transferred to Your Spouse\n\nQ",
    ),
    (None, "\n\n"),
    (
        "permission",
        "If you were born before January 1, {birth_year}, and you had a spouse on December 31, {year}, you and your spouse can jointly decide to include a portion\nof your retirement income in the calculation of your spouse’s income.",
    ),
    (None, " "),
    (
        "informational_guide",
        "For more information, see “Transfer of retirement income between spouses”\nin the instructions for line 122 in the guide.",
    ),
    (None, "\nNOTE\n"),
    (
        "corresponding_withholding",
        "If you transfer a portion of your retirement income to your spouse, you must also transfer the corresponding Québec income tax withheld from that income.",
    ),
    (None, "\n\n"),
    (
        "transferor_enclosure",
        "Only the person transferring retirement income must enclose Schedule Q with their return.",
    ),
    (None, "\n"),
    ("spouse_sin", "Your spouse’s\nsocial insurance number"),
    (None, "\n\n"),
    ("spouse_name", "Last name and first name of your spouse on December 31, {year}"),
    (None, "\n3\n\nA\n\n4\n\nRetirement income transferred to your spouse\n\n"),
    ("entered_income", "Amount from line 122 of your return."),
    (None, " "),
    (
        "compensation_guide",
        "If you received amounts under a retirement compensation arrangement,\nsee “Eligible retirement income” in the instructions for line 122 in the guide.",
    ),
    (None, "\n"),
    (
        "deduction12",
        "Amount from line 10 used to purchase an annuity or transferred to an RRSP, a RRIF\nor a PRPP/VRSP (included in the amount on line 250 [point 4])",
    ),
    (None, "\n12\n"),
    (
        "deduction13",
        "Deduction claimed on line 250 (point 6) for a refund of unused PRPP/VRSP\ncontributions included in the amount on line 10 above",
    ),
    (None, "\n+ 13\n"),
    (
        "deduction14",
        "Deduction claimed on line 293 of your return for the amount on line 10",
    ),
    (None, "\n+ 14\n"),
    (
        "deduction16",
        "Deduction claimed on line 297 (point 12) of your return for the amount on line 10",
    ),
    (None, "\n+ 16\n"),
    ("deductions_total", "Add lines 12 through 16."),
    (None, "\n=\n"),
    ("net_income", "Subtract line 18 from line 10."),
    (None, "\n"),
    (
        "selected_transfer",
        "Retirement income you are transferring to your spouse on December 31, {year} (maximum: 50% of the amount on line 20).",
    ),
    (None, "\n"),
    ("transferor_income_destination", "Carry this amount to line 245 of your return."),
    (None, "\n"),
    (
        "recipient_income_destination",
        "Your spouse must enter the same amount on line 123 of their return.",
    ),
    (
        None,
        "\n\nB\n\n10\n\n18\n\n= 20\n\n22\n\nQuébec income tax withholding transferred to your spouse\n\n",
    ),
    (
        "entered_rl2_withholding",
        "Québec income tax withheld (RL-2 slip, box J ) from the income on line 20.",
    ),
    (None, "\n"),
    (
        "conditional_rl1_withholding",
        "If the income includes amounts received under a retirement compensation arrangement,\ninclude, on this line, the amount shown in box E of your RL-1 slip.",
    ),
    (None, "\n"),
    ("withholding_numerator", "Amount from line 22"),
    (None, "\n"),
    ("withholding_denominator", "Amount from line 20"),
    (None, "\n"),
    (
        "withholding_proportion",
        "Multiply line 50 by line 52. Divide the result by line 53.",
    ),
    (None, "\n"),
    (
        "transferor_withholding_destination",
        "Carry this amount to line 451.1 of your return.",
    ),
    (None, "\n"),
    (
        "recipient_withholding_destination",
        "Your spouse must enter the same amount on line 451.3 of their return.",
    ),
    (None, "\n\n"),
    ("enclosure", "Enclose this schedule with your return."),
    (None, "\n"),
    ("copy_guide", "If you want to keep a copy for your records, see the guide."),
    (None, "\n\n50\n52\n53\n\n58\n\nY4Q1 ZZ 89528149\n\nPrescribed form\n\nQ"),
)


def _literal(text: str) -> str:
    return re.escape(text).replace("\\" + "\n", r"\r?\n")


def _pattern() -> re.Pattern[str]:
    seen: set[str] = set()

    def template(text: str) -> str:
        result = []
        for part in re.split(r"(\{(?:year|birth_year)\})", text):
            if part in {"{year}", "{birth_year}"}:
                key = part[1:-1]
                result.append(
                    rf"(?P={key})" if key in seen else rf"(?P<{key}>(?:19|20)\d{{2}})"
                )
                seen.add(key)
            else:
                result.append(_literal(part))
        return "".join(result)

    wrapper = (
        r"(?P<wrapper>Revenu Quebec Schedule Q (?P<wrapper_year>(?:19|20)\d{2})"
        + _literal(" - Retirement Income Transferred to Your Spouse\n\n")
        + r")?"
    )
    body = wrapper
    for role, text in _PARTS:
        pattern = template(text)
        body += rf"(?P<{role}>{pattern})" if role is not None else pattern
    return re.compile(r"\A" + body + r"\Z")


_DOCUMENT = _pattern()

# Each number has a printed role, not an automatic scalar-credit entitlement.
_NUMERIC_ROLES = {
    "header": ("form_identifier", "form_year", "form_revision_month"),
    "permission": (
        "birth_cutoff_day",
        "birth_cutoff_year",
        "spouse_reference_day",
        "spouse_reference_year",
    ),
    "informational_guide": ("external_return_line",),
    "spouse_name": ("spouse_reference_day", "spouse_reference_year"),
    "entered_income": ("external_return_line",),
    "compensation_guide": ("external_return_line",),
    "deduction12": (
        "included_income_row",
        "external_return_line",
        "external_return_point",
    ),
    "deduction13": (
        "external_return_line",
        "external_return_point",
        "included_income_row",
    ),
    "deduction14": ("external_return_line", "included_income_row"),
    "deduction16": (
        "external_return_line",
        "external_return_point",
        "included_income_row",
    ),
    "deductions_total": ("range_start_row", "range_end_row"),
    "net_income": ("subtrahend_row", "minuend_row"),
    "selected_transfer": (
        "spouse_reference_day",
        "spouse_reference_year",
        "cap_percent",
        "cap_base_row",
    ),
    "transferor_income_destination": ("external_return_line",),
    "recipient_income_destination": ("external_return_line",),
    "entered_rl2_withholding": ("slip_number", "income_row"),
    "conditional_rl1_withholding": ("slip_number",),
    "withholding_numerator": ("provider_row",),
    "withholding_denominator": ("provider_row",),
    "withholding_proportion": ("multiplicand_row", "multiplier_row", "divisor_row"),
    "transferor_withholding_destination": ("external_return_line",),
    "recipient_withholding_destination": ("external_return_line",),
}


@dataclass(frozen=True)
class TransferSourceRecord:
    role: str
    span: Span


@dataclass(frozen=True)
class PrintedNumber:
    owner: str
    role: str
    text: str
    span: Span


@dataclass(frozen=True)
class RetirementTransferSource:
    source: str
    citation: str
    source_sha256: str
    year: int
    birth_cutoff: date
    spouse_reference_date: date
    cap_percent: Decimal
    records: tuple[TransferSourceRecord, ...]
    numbers: tuple[PrintedNumber, ...]
    unowned_spans: tuple[Span, ...]

    def bound(self, source: str, citation: str) -> bool:
        return (
            source == self.source
            and citation == self.citation
            and recognize_retirement_transfer_schedule(source, citation) == self
        )


def recognize_retirement_transfer_schedule(
    source: str, citation: str
) -> RetirementTransferSource | None:
    """Recognize this whole form; no policy, witness, or factual authority."""
    if (
        not isinstance(source, str)
        or len(source) > 20_000
        or not isinstance(citation, str)
        or re.fullmatch(r"[a-z0-9-]+(?:/[a-z0-9-]+)+", citation) is None
    ):
        return None
    match = _DOCUMENT.fullmatch(source)
    if match is None:
        return None
    year = int(match["year"])
    if match["wrapper_year"] is not None and int(match["wrapper_year"]) != year:
        return None
    records = tuple(
        TransferSourceRecord(role, match.span(role))
        for role, _text in _PARTS
        if role is not None
    )
    numbers = []
    for record in records:
        start, end = record.span
        found = list(re.finditer(r"\d+(?:\.\d+)?", source[start:end]))
        roles = _NUMERIC_ROLES.get(record.role, ())
        if len(found) != len(roles):
            return None
        for token, role in zip(found, roles, strict=True):
            numbers.append(
                PrintedNumber(
                    record.role,
                    role,
                    token[0],
                    (start + token.start(), start + token.end()),
                )
            )
    # Detached locators are raw display references, not computed operands.
    end = 0
    gaps = []
    for record in records:
        if end < record.span[0]:
            gaps.append((end, record.span[0]))
        end = record.span[1]
    if end < len(source):
        gaps.append((end, len(source)))
    return RetirementTransferSource(
        source,
        citation,
        hashlib.sha256(source.encode()).hexdigest(),
        year,
        date(int(match["birth_year"]), 1, 1),
        date(year, 12, 31),
        Decimal(next(n.text for n in numbers if n.role == "cap_percent")),
        records,
        tuple(numbers),
        tuple(gaps),
    )
