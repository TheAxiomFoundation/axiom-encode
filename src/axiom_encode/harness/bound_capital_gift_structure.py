"""Closed 2025 T1170 raw-source records; no execution or coverage authority.

The resolver's rotated table extraction is retained verbatim. This module does
not reverse source text, admit evidence, bind formulas, or waive any findings.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

Span = tuple[int, int]
_CITATION = "ca/policy/cra/t1-2025/capital-gains-on-gifts"

_RECORD_TEXT: tuple[tuple[str, str], ...] = (
    (
        "resolver_title",
        "CRA Form T1170 2025 Capital Gains on Gifts of Certain Capital Property",
    ),
    ("physical_title", "Capital Gains on Gifts of Certain Capital Property"),
    (
        "property_scope",
        "Use this form if you donated any of the following types of properties to a registered charity or other qualified donee in 2025:\n(cid:129) a share, debt obligation, or right listed on a designated stock exchange\n(cid:129) a share of the capital stock of a mutual fund corporation\n(cid:129) a unit of a mutual fund trust\n(cid:129) a prescribed debt obligation\n(cid:129) an interest in a related segregated fund trust\n(cid:129) ecologically sensitive land (including a covenant, an easement or, for land in Quebec, a real servitude or, under certain\nconditions, a personal servitude) where it is gifted to certain qualified donees",
    ),
    (
        "gift_inclusion",
        "If there is no advantage in respect of the gift, the full amount of the capital gain realized on the gift is eligible for an inclusion\nrate of zero. However, if there is an advantage, only a portion of the capital gain is eligible for the inclusion rate of zero. The\nrest is subject to the inclusion rate of 50%.",
    ),
    (
        "prior_exchange",
        "The inclusion rate of zero is extended to any capital gain realized on the exchange of shares of the capital stock of a\ncorporation for securities listed in the first five bullets above that are donated within 30 days of the exchange if certain\nconditions are met.",
    ),
    (
        "partnership",
        "If the exchanged property is a partnership interest (other than prescribed interests in a partnership), a special calculation is\nrequired to determine what the capital gain will be. This amount should be reported directly on line 17400 in Part 4 of\nSchedule 3, Capital Gains or Losses.",
    ),
    (
        "flow_through_scope",
        "You may donate property to a qualified donee that is at the time of the donation included in a flow-through share (FTS) class\nof property. At that time, you may also have an exemption threshold in respect of the FTS class of property. If this is the case,\nyou may be deemed to have an additional capital gain from the disposition of another capital property subject to the inclusion\nrate calculated on Schedule 3. This additional capital gain is used in the calculation of adjusted taxable income for purposes\nof the alternative minimum tax.",
    ),
    (
        "credit_and_guide_reference",
        "You or your spouse or common-law partner may be able to claim a non-refundable tax credit for these gifts.\nFor more information, see Guide P113, Gifts and Income Tax.",
    ),
    (
        "page_one_display",
        "T1170 E (25) (Ce formulaire est disponible en français.) Page 1 of 2\n\ndetelpmoc\nnehw\nB\ndetcetorP\nytreporP\nlatipaC\nniatreC\nfo\nstfiG\nno\nsniaG\nlatipaC",
    ),
    (
        "document_duties",
        ".nruter\nruoy\not\nrepap\nfo\nteehs\netarapes\na\nhcatta\n,ecaps\nerom\ndeen\nuoy\nfI\n.nruter\nruoy\nhtiw\nmrof\nsiht\nedulcnI",
    ),
    ("disposition_year", "5202\nni\nsnoitisopsiD"),
    (
        "shares_table",
        "stinu\ndnuf\nlautum\ndna\nstinu\ndnuf\ndetagerges\n,serahs\nrehto\n,dedart\nylcilbup\nera\ntaht\nsthgir\ndna\nserahS\n)8(\n)7(\n%05\not\ntcejbus\nniaG\nrof\nelbigile\nniaG\netar\nnoisulcni\netar\nnoisulcni\n%0\n)5(\n)6(\n)3(\n)2(\n)1(\n5\nnmuloc(\ndeilpitlum\n5\nnmuloc(\n)4(\nnoitaroproc\nro\ndnuf\nfo emaN\nniaG\ntnuoma\nelbigilE\ndetsujdA\nsdeecorP\nraeY\nrebmuN\nsunim\ndedivid\n6\nnmuloc\nyb\nsyaltuO\nserahs\nfo\nssalc\ndna\nsunim\n2\nnmuloc(\ntfig\nfo\nesab\ntsoc\nnoitisopsid\nfo\nderiuqca\n)7\nnmuloc\n)2\nnmuloc\nyb\nsesnepxe\ndna\n)4\ndna\n3\nsnmuloc\n+\n+\n+\n+\n+\n+\n=\n=\n=\n1\n03286\nlatoT\n02286\nlatoT\n.3\neludehcS\nfo\n3\ntraP\nni\n00231\nenil\nno\n8\nnmuloc\nfo\n1\nenil\nmorf\ntnuoma\neht\nedulcnI",
    ),
    (
        "bonds_table",
        "seitreporp\nrehto\ndna\n,dedart\nylcilbup\nera\ntaht\nserutnebed\ndna\nsdnoB\n)8(\n)7(\n%05\not\ntcejbus\nniaG\nrof\nelbigile\nniaG\netar\nnoisulcni\netar\nnoisulcni\n%0\n)5(\n)6(\n)3(\n)2(\n)1(\n5\nnmuloc(\ndeilpitlum\n5\nnmuloc(\n)4(\netad\nytirutaM\nniaG\ntnuoma\nelbigilE\ndetsujdA\nsdeecorP\nraeY\nreussi\nfo\nemaN\neulav\necaF\nsunim\ndedivid\n6\nnmuloc\nyb\nsyaltuO\n)DDMMYYYY(\nsunim\n2\nnmuloc(\ntfig\nfo\nesab\ntsoc\nnoitisopsid\nfo\nderiuqca\n)7\nnmuloc\n)2\nnmuloc\nyb\nsesnepxe\ndna\n)4\ndna\n3\nsnmuloc\n+\n+\n+\n+\n+\n+\n=\n=\n=\n2\n05286\nlatoT\n04286\nlatoT\n.3\neludehcS\nfo\n3\ntraP\nni\n00351\nenil\nno\n8\nnmuloc\nfo\n2\nenil\nmorf\ntnuoma\neht\nedulcnI",
    ),
    (
        "land_table",
        ",ro\nedutivres\nlaer\na\n,cebeuQ\nni\ndnal\nrof\n,ro\ntnemesae\nna\n,tnanevoc\na\ngnidulcni(\ndnal\nevitisnes\nyllacigolocE\n)8(\n)7(\n)edutivres\nlanosrep\na\n,snoitidnoc\nniatrec\nrednu\n%05\not\ntcejbus\nniaG\nrof\nelbigile\nniaG\netar\nnoisulcni\netar\nnoisulcni\n%0\n)5(\n)6(\n)3(\n)2(\n)1(\n5\nnmuloc(\ndeilpitlum\n5\nnmuloc(\n)4(\nniaG\ntnuoma\nelbigilE\ndetsujdA\nsdeecorP\nraeY\ndnal\nfo\nnoitpircsed\nlagel\nro\nsserddA\nsunim\ndedivid\n6\nnmuloc\nyb\nsyaltuO\nsunim\n2\nnmuloc(\ntfig\nfo\nesab\ntsoc\nnoitisopsid\nfo\nderiuqca\n)7\nnmuloc\n)2\nnmuloc\nyb\nsesnepxe\ndna\n)4\ndna\n3\nsnmuloc\n+\n+\n+\n+\n+\n+\n=\n=\n=\n3\n25286\nlatoT\n15286\nlatoT\n.3\neludehcS\nfo\n3\ntraP\nni\n00831\nenil\nno\n8\nnmuloc\nfo\n3\nenil\nmorf\ntnuoma\neht\nedulcnI",
    ),
    (
        "flow_through_route",
        "3\neludehcS\nno\ndedulcni\nytreporp\nfo\nssalc\n)STF(\nerahs\nhguorht-wolf\nfo\nnoitanod\neht\nno\nsniag\nlatipac\ndemeed\nfo\nlatoT\n)xaT\nemocnI\ndna\nstfiG\n,311P\nediuG\nees(\n4\n13286\n.xaT\nmuminiM\nevitanretlA\n,196T\nmroF\nfo\n1\ntraP\nfo\n52\nenil\nno\n4\nenil\nmorf\ntnuoma\neht\nedulcnI",
    ),
    (
        "page_two_display",
        ".nruter\nruoy\nno\neciton\nycavirp\neht\neeS\n2\nfo\n2\negaP\n)52(\nE\n0711T",
    ),
)


@dataclass(frozen=True)
class CapitalGiftRecord:
    role: str
    span: Span


@dataclass(frozen=True)
class CapitalGiftTable:
    role: str
    span: Span
    columns: tuple[CapitalGiftRecord, ...]
    references: tuple[CapitalGiftRecord, ...]
    destination: str
    proceeds_field: str
    zero_inclusion_field: str


@dataclass(frozen=True)
class CapitalGiftSource:
    source: str
    citation: str
    source_sha256: str
    year: int
    records: tuple[CapitalGiftRecord, ...]
    subrecords: tuple[CapitalGiftRecord, ...]
    tables: tuple[CapitalGiftTable, ...]
    unowned_spans: tuple[Span, ...]

    def bound(self, source: str, citation: str) -> bool:
        """Re-recognize source identity; this does not certify a consumer."""
        return recognize_capital_gift_source(source, citation) == self


def _record(role: str, text: str, needle: str, offset: int) -> CapitalGiftRecord:
    # All callers supply a closed literal record, so uniqueness is structural.
    if text.count(needle) != 1:
        raise ValueError(f"nonunique source component: {role}")
    start = offset + text.index(needle)
    return CapitalGiftRecord(role, (start, start + len(needle)))


def _table(role: str, text: str, offset: int) -> CapitalGiftTable:
    destination, proceeds, zero = {
        "shares_table": ("13200", "68220", "68230"),
        "bonds_table": ("15300", "68240", "68250"),
        "land_table": ("13800", "68251", "68252"),
    }[role]
    columns = tuple(
        _record(f"column_{number}", text, f"){number}(", offset)
        for number in range(1, 9)
    )
    repeated_fives = tuple(re.finditer(re.escape("5\nnmuloc("), text))
    if len(repeated_fives) != 2:
        raise ValueError("unrecognized ordered column references")
    references = [
        CapitalGiftRecord(label, (offset + match.start(), offset + match.end()))
        for label, match in zip(
            ("column_8_minuend_column_5", "column_7_multiplicand_column_5"),
            repeated_fives,
            strict=True,
        )
    ]
    for label, needle in (
        ("column_7_multiplier_column_6", "6\nnmuloc\nyb"),
        ("column_7_divisor_column_2", ")2\nnmuloc\nyb"),
        ("column_8_subtrahend_column_7", ")7\nnmuloc"),
        ("column_5_minuend_column_2", "2\nnmuloc("),
        ("column_5_deductions_columns_3_4", ")4\ndna\n3\nsnmuloc"),
        ("zero_rate", "etar\nnoisulcni\n%0"),
        ("half_rate", "%05"),
        ("destination", destination[::-1]),
        ("proceeds_field", proceeds[::-1]),
        ("zero_inclusion_field", zero[::-1]),
    ):
        references.append(_record(label, text, needle, offset))
    return CapitalGiftTable(
        role,
        (offset, offset + len(text)),
        columns,
        tuple(references),
        destination,
        proceeds,
        zero,
    )


def recognize_capital_gift_source(
    source: str, citation: str = _CITATION
) -> CapitalGiftSource | None:
    """Recognize only the complete, fixed-2025 resolver body and raw layout.

    Literal source records deliberately refuse other editions and extraction
    layouts. Separators are matched exactly and remain unowned. Table children
    overlap their parent records; only top-level records partition the source.
    """
    if not isinstance(source, str) or citation != _CITATION or len(source) > 20_000:
        return None
    records: list[CapitalGiftRecord] = []
    unowned: list[Span] = []
    cursor = 0
    for index, (role, text) in enumerate(_RECORD_TEXT):
        # The title has a blank separator; every subsequent record has one LF.
        separator = "" if index == 0 else "\n\n" if index == 1 else "\n"
        if not source.startswith(separator, cursor):
            return None
        if separator:
            unowned.append((cursor, cursor + len(separator)))
            cursor += len(separator)
        if not source.startswith(text, cursor):
            return None
        records.append(CapitalGiftRecord(role, (cursor, cursor + len(text))))
        cursor += len(text)
    if cursor != len(source):
        return None
    subrecords: list[CapitalGiftRecord] = []
    tables: list[CapitalGiftTable] = []
    for record in records:
        start, end = record.span
        text = source[start:end]
        if record.role.endswith("_table"):
            tables.append(_table(record.role, text, start))
        elif record.role == "property_scope":
            bullets = tuple(re.finditer(re.escape("(cid:129) "), text))
            for index, bullet in enumerate(bullets):
                stop = (
                    bullets[index + 1].start()
                    if index + 1 < len(bullets)
                    else len(text)
                )
                subrecords.append(
                    CapitalGiftRecord(
                        f"property_alternative_{index + 1}",
                        (start + bullet.start(), start + stop),
                    )
                )
        elif record.role == "gift_inclusion":
            advantage = text.index("However, if there is an advantage")
            remainder = text.index("The\nrest is subject")
            for label, left, right in (
                ("no_advantage", 0, advantage),
                ("advantage", advantage, remainder),
                ("half_inclusion", remainder, len(text)),
            ):
                subrecords.append(
                    CapitalGiftRecord(label, (start + left, start + right))
                )
        elif record.role == "document_duties":
            enclosure = text.index(".nruter\nruoy\nhtiw")
            subrecords.extend(
                (
                    CapitalGiftRecord("additional_sheet", (start, start + enclosure)),
                    CapitalGiftRecord("enclosure", (start + enclosure, end)),
                )
            )
    return CapitalGiftSource(
        source,
        citation,
        hashlib.sha256(source.encode()).hexdigest(),
        2025,
        tuple(records),
        tuple(subrecords),
        tuple(tables),
        tuple(unowned),
    )
