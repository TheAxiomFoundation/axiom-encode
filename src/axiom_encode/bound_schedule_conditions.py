"""Closed raw Schedule condition records; no input or execution authority."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from axiom_encode.bound_schedule_structure import BoundScheduleStructure, Span

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MEDICAL = re.compile(
    r"Line (?P<field>[1-9]\d{4}): If, on "
    r"(?P<date>(?P<month>"
    + "|".join(_MONTHS)
    + r") (?P<day>\d{1,2}), (?P<year>\d{4})), "
    r"(?P<relationship>you and your spouse or common-law\s+partner) "
    r"(?P<residence>occupied separate principal residences) "
    r"(?P<reason>for medical reasons), you can\s+both claim "
    r"(?P<benefit>[A-Za-z' \r\n-]{3,160})\. "
    r"(?P<reporting>Enter your spouse's or\s+common-law partner's address on line "
    r"(?P=field)\.) \^"
)
_RESIDENCE = re.compile(
    r"(?m)^Do the expenses on line (?P<row>[1-9]\d{0,3}) relate to the residence located at the address you\r?\n"
    r"entered on page (?P<page>[1-9]\d?) of your return\? "
    r"(?P<yes>If yes, tick this box\. \^)\r?\n\r?\n"
    r"(?P<no>If not, enter the address of the residence they relate to: \^)\r?$"
)
_OPERATIVE = re.compile(
    r"\b(?:if|unless|except|provided|subject|but|however|not|never|otherwise|"
    r"when|plus|minus|less|subtract|add|multiply|divide|maximum|minimum|"
    r"claim|ignore|apply|reduce|increase|pay|owe|require|must|may|can|shall)\b",
    re.I,
)


@dataclass(frozen=True)
class MedicalSeparationRecord:
    span: Span
    reference_date_span: Span
    reference_date: date
    relationship_span: Span
    separate_residences_span: Span
    medical_reason_span: Span
    permission_span: Span
    address_reporting_span: Span
    reporting_field: int


@dataclass(frozen=True)
class ResidenceReportingRecord:
    span: Span
    yes_span: Span
    no_span: Span
    expense_row: int
    return_page: int


@dataclass(frozen=True)
class ScheduleConditionRecords:
    form: BoundScheduleStructure
    medical: MedicalSeparationRecord
    residence: ResidenceReportingRecord

    def bound(self, source: str, citation: str) -> bool:
        return (
            self.form.bound(source, citation)
            and recognize_schedule_conditions(self.form) == self
        )


def _closed_context(source: str, span: Span) -> bool:
    """A plain preceding heading cannot negate/qualify the following paragraph."""
    before = source[: span[0]].rstrip()
    paragraphs = re.split(r"\r?\n\r?\n", before)
    heading = paragraphs[-1]
    enclosing = paragraphs[-2].strip() if len(paragraphs) > 1 else ""
    after = source[span[1] :]
    if after and not re.match(r"\r?\n\r?\n", after):
        return False
    return bool(
        re.fullmatch(r"[A-Za-z][A-Za-z '()-]{2,100}", heading)
        and not _OPERATIVE.search(heading)
        and not enclosing.endswith(":")
        and not re.match(
            r"(?:do\s+not|never|if|when|unless|except|provided|subject\s+to)\b",
            enclosing,
            re.I,
        )
        and not re.match(
            r"\s*(?:do\s+not|never|if|when|unless|except|provided|subject\s+to|but|however)\b",
            after,
            re.I,
        )
    )


def recognize_schedule_conditions(
    form: BoundScheduleStructure,
) -> ScheduleConditionRecords | None:
    """Keep antecedent, permission and reporting effects separately addressable.

    These records do not bind caller facts, formulas, source gates or snapshots.
    In particular, a false exception predicate never means general ineligibility.
    """
    if not isinstance(form, BoundScheduleStructure) or not form.bound(
        form.source, form.citation
    ):
        return None
    source = form.source
    medicals = []
    for field in form.reporting_fields:
        raw = source[slice(*field.span)]
        match = _MEDICAL.fullmatch(raw)
        if match is None:
            continue
        if not _closed_context(source, field.span) or _OPERATIVE.search(
            match["benefit"]
        ):
            return None
        try:
            reference_date = date(
                int(match["year"]), _MONTHS.index(match["month"]) + 1, int(match["day"])
            )
        except ValueError:
            return None
        if reference_date.year != form.year or int(match["field"]) != field.number:
            return None

        def absolute(group: str) -> Span:
            a, b = match.span(group)
            return field.span[0] + a, field.span[0] + b

        medicals.append(
            MedicalSeparationRecord(
                field.span,
                absolute("date"),
                reference_date,
                absolute("relationship"),
                absolute("residence"),
                absolute("reason"),
                (field.span[0] + raw.index("you can"), absolute("benefit")[1] + 1),
                absolute("reporting"),
                field.number,
            )
        )
    residence = tuple(_RESIDENCE.finditer(source))
    if len(medicals) != 1 or len(residence) != 1:
        return None
    match = residence[0]
    if (
        not _closed_context(source, match.span())
        or int(match["row"]) != form.rows[0].number
    ):
        return None
    if not medicals[0].span[1] < match.start():
        return None
    # Both records must remain in the source-only structure's unowned regions.
    if any(
        not any(a <= start < end <= b for a, b in form.unowned_spans)
        for start, end in (medicals[0].span, match.span())
    ):
        return None
    return ScheduleConditionRecords(
        form,
        medicals[0],
        ResidenceReportingRecord(
            match.span(),
            match.span("yes"),
            match.span("no"),
            int(match["row"]),
            int(match["page"]),
        ),
    )
