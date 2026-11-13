"""Closed whole-form raw records; no factual, execution or coverage authority."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

Span = tuple[int, int]
_PARTS = (
    (
        "header",
        "\ufeff5003-TC\n\nProtected B when completed\n\n{year}\n\nForm NS479\n\nNova Scotia Credits",
    ),
    (
        "refund",
        "You can claim these refundable tax credits even if you do not have to pay any\ntax. If the total of these credits is more than the amount of taxes you have\nto pay, you may get a refund for the difference.",
    ),
    (
        "paper_attachment",
        "Complete the calculations that apply to you and attach a copy of this form to\nyour paper return.",
    ),
    (
        "eligibility_referral",
        "For more information about these credits, including credit eligibility\nrequirements, go to canada.ca/ns-tax-info.",
    ),
    (
        "volunteer_heading",
        "Volunteer firefighters and ground search and rescue tax credit",
    ),
    (
        "volunteer_amount",
        "Line 1 and line 62400: Volunteer firefighters and ground search and rescue\ntax credit Claim $500 ^",
    ),
    ("children_heading", "Children's sports and arts tax credit"),
    (
        "table_applicability_and_assistance",
        "If applicable, complete the following table and include all of your\nchildren's expenses for programs of artistic, cultural, recreational, or\ndevelopmental activity or physical activity. Assistance includes\nreimbursements, allowances, and any form of assistance received or\nreceivable.",
    ),
    (
        "receipt_retention_and_non_enclosure",
        "Do not include your receipts but keep them in case you are asked to provide\nthem later.",
    ),
    (
        "additional_sheet",
        "If you need more space, attach an additional sheet of paper.",
    ),
    (
        "transcriber_table",
        "*** Transcriber's Note: In print, the following table is set up as six\ncolumns with the possibility of six entries. The columns are labelled as:\nColumn A: Child's name (one child per line); Column B: All eligible expenses\nper child for the year; Column C: Assistance received or receivable for\namount in column B; Column D: Column B minus column C (Maximum $500 per\nchild); Column E: Amounts from column D claimed by others; Column F: Column D\nminus column E. ***",
    ),
    ("column_a", "Column A Child's name (one child per line) ^"),
    ("column_b", "Column B All eligible expenses per child for the year ^"),
    ("column_c", "Column C Assistance received or receivable for amount in column B ^"),
    ("column_d", "Column D Column B minus column C (Maximum $500 per child) ^"),
    ("column_e", "Column E Amounts from column D claimed by others ^"),
    ("column_f", "Column F Column D minus column E ^"),
    (
        "line2_sum_f",
        "Line 2 and line 62410: Your children's sports and arts expenses Total of all\namounts in column F of the table above = ^",
    ),
    ("line3_rate", "Line 3: Applicable rate 8.79%"),
    (
        "line4_product",
        "Line 4: Line 2 multiplied by the percentage from line 3 Children's sports and\narts tax credit = ^",
    ),
    (
        "line5_sum_destination",
        "Line 5: Line 1 plus line 4. Enter this amount on line 47900 of your return.\nNova Scotia Credits = ^",
    ),
    ("privacy_referral", "See the privacy notice on your return."),
)


@dataclass(frozen=True)
class ChildCreditRecord:
    role: str
    span: Span


@dataclass(frozen=True)
class ChildCreditForm:
    source: str
    citation: str
    source_sha256: str
    year: int
    records: tuple[ChildCreditRecord, ...]
    unowned_spans: tuple[Span, ...]

    def bound(self, source: str, citation: str) -> bool:
        return (
            source == self.source
            and citation == self.citation
            and recognize_child_credit_form(source, citation) == self
        )


def _literal(text: str) -> str:
    return re.escape(text).replace("\\" + "\n", r"\r?\n")


def _pattern() -> re.Pattern[str]:
    wrapper = (
        r"(?P<wrapper>CRA Form NS479 (?P<wrapper_year>(?:19|20)\d{2}) Nova Scotia Credits"
        + _literal("\n\n")
        + ")?"
    )
    parts = []
    for role, literal in _PARTS:
        pattern = _literal(literal).replace(r"\{year\}", r"(?P<year>(?:19|20)\d{2})")
        parts.append(rf"(?P<{role}>{pattern})")
    return re.compile(r"\A" + wrapper + _literal("\n\n").join(parts) + r"\Z")


_DOCUMENT = _pattern()


def recognize_child_credit_form(source: str, citation: str) -> ChildCreditForm | None:
    """Retain the full form, including procedural/external obligations and raw gaps.

    Literal layout six entries is not a legal child limit. These records grant no
    numeric recall, formula, condition, snapshot or related-provider authority.
    """
    if (
        not isinstance(source, str)
        or len(source) > 20000
        or not isinstance(citation, str)
        or re.fullmatch(r"[a-z0-9-]+(?:/[a-z0-9-]+)+", citation) is None
    ):
        return None
    match = _DOCUMENT.fullmatch(source)
    if match is None or (
        match["wrapper_year"] is not None and match["wrapper_year"] != match["year"]
    ):
        return None
    records = tuple(ChildCreditRecord(role, match.span(role)) for role, _ in _PARTS)
    gaps = []
    end = 0
    for record in records:
        if end < record.span[0]:
            gaps.append((end, record.span[0]))
        end = record.span[1]
    if end < len(source):
        gaps.append((end, len(source)))
    return ChildCreditForm(
        source,
        citation,
        hashlib.sha256(source.encode()).hexdigest(),
        int(match["year"]),
        records,
        tuple(gaps),
    )
