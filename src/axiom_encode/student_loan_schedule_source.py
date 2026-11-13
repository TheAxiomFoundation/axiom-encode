"""Closed raw student-loan schedule records, without execution authority.

Recognition preserves the entire document. It neither credits numeric values
nor admits a snapshot, proves a formula, or discharges a source condition.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

Span = tuple[int, int]


def _literal(text: str) -> str:
    return re.escape(text).replace("\\" + "\n", r"\r?\n")


_RECORDS = (
    (
        "current_completion",
        "Complete this schedule if interest was paid in {year} on a student loan granted to you.",
    ),
    (
        "prior_completion",
        "Also complete this schedule if, for {prior}, you calculated an amount for interest paid on a student loan and decided to carry the amount forward\nto a future year.",
    ),
    (
        "historical_reconstruction",
        "If interest was paid on your student loan after {cutoff}, but in {prior} you did not calculate the amount that could be carried forward, enter the result\nof the following calculation on line 46: interest paid for {first} through {prior} minus the interest already used to calculate a credit for interest paid\non a student loan for previous years.",
    ),
    (
        "entered_prior",
        "Unused portion of the interest paid on a student loan (line 62 of Schedule M for {prior}\nor the amount shown in this respect on your {prior} notice of assessment)",
    ),
    ("current_payment", "Interest paid in {year} on a student loan\n+"),
    (
        "available",
        "Add lines 46 and 48.\nCarry all or part of this amount to line 385 of your return.\x08\nInterest paid on a student loan =",
    ),
    ("return_claim", "Amount carried to line 385 of your {year} return\n–"),
    (
        "carryforward",
        "Subtract line 60 from line 52.\n\x08\nAmount that can be carried forward to a future year =",
    ),
    ("enclosure", "Enclose this schedule with your return."),
    ("guide_referral", "If you want to keep a copy for your records, see the guide."),
)


def _pattern() -> re.Pattern[str]:
    seen: set[str] = set()

    def template(text: str) -> str:
        parts = re.split(r"(\{(?:year|prior|cutoff|first)\})", text)
        result = []
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                key = part[1:-1]
                result.append(
                    rf"(?P={key})" if key in seen else rf"(?P<{key}>(?:19|20)\d{{2}})"
                )
                seen.add(key)
            else:
                result.append(_literal(part))
        return "".join(result)

    # Optional resolver title must agree with the physical document year/title.
    wrapper = (
        r"(?P<wrapper>Revenu Quebec Schedule M (?P<wrapper_year>(?:19|20)\d{2})"
        + _literal(" - Interest Paid on a Student Loan\n\n")
        + r")?"
    )
    header = template(
        "T P -1. D. M -V ({year}-12)\nT\n\nInterest Paid on a Student Loan\n\nSCHEDULE\n\nM"
    )
    body = wrapper + rf"(?P<header>{header})" + _literal("\n\n")
    for index, (name, text) in enumerate(_RECORDS):
        body += rf"(?P<{name}>{template(text)})"
        body += _literal("\n\n" if index in {7, 9} else "\n")
    # Printer identifier and detached locators are display records, not operands.
    body += r"(?P<printer>[A-Z]\d[A-Z]\d [A-Z]{2} \d{8})"
    body += _literal("\n\n") + r"(?P<locators>" + _literal("46\n48\n52\n60\n62") + r")"
    body += _literal("\n\nM")
    return re.compile(r"\A" + body + r"\Z")


_DOCUMENT = _pattern()


@dataclass(frozen=True)
class SourceRecord:
    role: str
    span: Span


@dataclass(frozen=True)
class StudentLoanScheduleSource:
    source: str
    citation: str
    source_sha256: str
    year: int
    prior_year: int
    history_first_year: int
    history_cutoff_year: int
    records: tuple[SourceRecord, ...]
    # Display and separators remain explicit, rather than silently discarded.
    unowned_spans: tuple[Span, ...]

    def bound(self, source: str, citation: str) -> bool:
        return (
            source == self.source
            and citation == self.citation
            and recognize_student_loan_schedule(source, citation) == self
        )


def recognize_student_loan_schedule(
    source: str, citation: str
) -> StudentLoanScheduleSource | None:
    """Describe this complete closed source shape; never certify its consumer."""
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
    year, prior, cutoff, first = (
        int(match[k]) for k in ("year", "prior", "cutoff", "first")
    )
    if (
        prior != year - 1
        or first != cutoff + 1
        or not cutoff < first <= prior
        or (match["wrapper_year"] is not None and int(match["wrapper_year"]) != year)
    ):
        return None
    records = tuple(SourceRecord(name, match.span(name)) for name, _ in _RECORDS)
    unowned = []
    end = 0
    for record in records:
        if end < record.span[0]:
            unowned.append((end, record.span[0]))
        end = record.span[1]
    if end < len(source):
        unowned.append((end, len(source)))
    return StudentLoanScheduleSource(
        source,
        citation,
        hashlib.sha256(source.encode()).hexdigest(),
        year,
        prior,
        first,
        cutoff,
        records,
        tuple(unowned),
    )
