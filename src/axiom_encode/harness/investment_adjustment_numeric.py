"""Closed, occurrence-specific Schedule N numeric roles; no evidence authority."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .bound_investment_adjustment_operands import bind_investment_adjustment

# Independently reviewed full digit-run inventory of the authenticated parent.
# Coordinates distinguish identical spellings with different source roles.
_OWNERS = (
    (
        (25, 29),
        "2025",
        "closed_2025_version",
        ("all19staticversions", "allactual2025requests"),
    ),
    ((72, 73), "1", "noncomputational_identifier", ()),
    (
        (83, 87),
        "2025",
        "closed_2025_version",
        ("all19staticversions", "allactual2025requests"),
    ),
    ((88, 90), "12", "noncomputational_identifier", ()),
    ((132, 134), "10", "printed_row_graph_locator", (18,)),
    ((138, 140), "12", "printed_row_graph_locator", (18,)),
    ((143, 145), "14", "printed_row_graph_locator", (18,)),
    ((148, 150), "16", "printed_row_graph_locator", (18,)),
    ((154, 156), "18", "printed_row_graph_locator", (18, 40, 60)),
    ((228, 231), "128", "upstream_or_destination_locator", (20, 36)),
    ((291, 294), "130", "upstream_or_destination_locator", (22, 36)),
    ((392, 394), "29", "upstream_or_destination_locator", (24, 36)),
    ((420, 423), "136", "upstream_or_destination_locator", (24, 36)),
    ((489, 490), "3", "noncomputational_identifier", ()),
    ((545, 548), "154", "upstream_or_destination_locator", (28,)),
    ((556, 558), "11", "upstream_or_destination_locator", (28,)),
    (
        (588, 590),
        "50",
        "substantive_half_rate",
        ("resource_recovery_rate", "line28actualnonzero"),
    ),
    ((615, 616), "2", "noncomputational_identifier", ()),
    ((808, 811), "260", "upstream_or_destination_locator", (34, 36)),
    ((836, 838), "20", "printed_row_graph_locator", (36,)),
    ((847, 849), "34", "printed_row_graph_locator", (36,)),
    ((865, 867), "36", "printed_row_graph_locator", (36, 40, 60)),
    ((878, 880), "18", "printed_row_graph_locator", (18, 40, 60)),
    ((915, 916), "0", "substantive_zero_floor", (40,)),
    ((943, 946), "260", "upstream_or_destination_locator", (40,)),
    ((1071, 1073), "29", "upstream_or_destination_locator", (10, 18)),
    ((1099, 1102), "136", "upstream_or_destination_locator", (10, 18)),
    ((1164, 1167), "231", "upstream_or_destination_locator", (12, 18)),
    ((1245, 1248), "260", "upstream_or_destination_locator", (14, 18)),
    ((1321, 1324), "260", "upstream_or_destination_locator", (16, 18)),
    ((1349, 1351), "10", "printed_row_graph_locator", (18,)),
    ((1360, 1362), "16", "printed_row_graph_locator", (18,)),
    ((1378, 1380), "20", "printed_row_graph_locator", (36,)),
    ((1384, 1386), "22", "printed_row_graph_locator", (36,)),
    ((1389, 1391), "24", "printed_row_graph_locator", (36,)),
    ((1394, 1396), "26", "printed_row_graph_locator", (36,)),
    ((1399, 1401), "28", "printed_row_graph_locator", (28, 36)),
    ((1404, 1406), "30", "printed_row_graph_locator", (36,)),
    ((1409, 1411), "32", "printed_row_graph_locator", (36,)),
    ((1414, 1416), "34", "printed_row_graph_locator", (36,)),
    ((1420, 1422), "36", "printed_row_graph_locator", (36, 40, 60)),
    ((1460, 1462), "40", "printed_row_graph_locator", (40, 76)),
    ((1549, 1552), "289", "upstream_or_destination_locator", (50, 54)),
    ((1557, 1560), "290", "upstream_or_destination_locator", (52, 54)),
    ((1620, 1623), "289", "upstream_or_destination_locator", (50, 54)),
    ((1686, 1689), "276", "upstream_or_destination_locator", (52, 54, 64)),
    ((1697, 1698), "9", "upstream_or_destination_locator", (52, 54, 64)),
    ((1724, 1726), "50", "printed_row_graph_locator", (54,)),
    ((1731, 1733), "52", "printed_row_graph_locator", (54,)),
    ((1752, 1754), "36", "printed_row_graph_locator", (36, 40, 60)),
    ((1778, 1780), "18", "printed_row_graph_locator", (18, 40, 60)),
    ((1801, 1803), "58", "printed_row_graph_locator", (18, 60)),
    ((1814, 1816), "56", "printed_row_graph_locator", (36, 60)),
    ((1851, 1852), "0", "substantive_zero_floor", (60,)),
    ((1868, 1870), "60", "printed_row_graph_locator", (60, 64)),
    ((1881, 1883), "54", "printed_row_graph_locator", (54, 64)),
    ((1918, 1919), "0", "substantive_zero_floor", (64,)),
    ((1946, 1949), "276", "upstream_or_destination_locator", (64,)),
    ((1971, 1973), "50", "printed_row_graph_locator", (54,)),
    ((1977, 1979), "52", "printed_row_graph_locator", (54,)),
    ((1983, 1985), "54", "printed_row_graph_locator", (54, 64)),
    ((1987, 1989), "56", "printed_row_graph_locator", (36, 60)),
    ((1993, 1995), "58", "printed_row_graph_locator", (18, 60)),
    ((1996, 1998), "60", "printed_row_graph_locator", (60, 64)),
    ((2042, 2044), "64", "printed_row_graph_locator", (64, 76)),
    ((2148, 2151), "252", "upstream_or_destination_locator", (70, 76)),
    ((2183, 2185), "40", "printed_row_graph_locator", (40, 76)),
    ((2209, 2211), "64", "printed_row_graph_locator", (64, 76)),
    ((2228, 2230), "70", "printed_row_graph_locator", (76,)),
    ((2239, 2241), "74", "printed_row_graph_locator", (64, 76)),
    ((2260, 2263), "252", "upstream_or_destination_locator", (78,)),
    (
        (2272, 2276),
        "2025",
        "closed_2025_version",
        ("all19staticversions", "allactual2025requests"),
    ),
    ((2298, 2300), "78", "printed_row_graph_locator", (78, 80)),
    ((2311, 2313), "76", "printed_row_graph_locator", (76, 80)),
    ((2419, 2421), "70", "printed_row_graph_locator", (76,)),
    ((2432, 2434), "72", "printed_row_graph_locator", (40, 76)),
    ((2435, 2437), "74", "printed_row_graph_locator", (64, 76)),
    ((2438, 2440), "76", "printed_row_graph_locator", (76, 80)),
    ((2441, 2443), "78", "printed_row_graph_locator", (78, 80)),
    ((2482, 2484), "80", "printed_row_graph_locator", (80,)),
    ((2487, 2488), "4", "noncomputational_identifier", ()),
    ((2489, 2490), "1", "noncomputational_identifier", ()),
    ((2494, 2502), "89527849", "noncomputational_identifier", ()),
)


@dataclass(frozen=True)
class InvestmentAdjustmentNumericOwner:
    span: tuple[int, int]
    literal: str
    role: str
    owners: tuple


def investment_adjustment_numeric_owners(bound, content, parent, guide):
    """Return roles only for the full closed static source-and-formula binding."""
    current = bind_investment_adjustment(content, parent, guide)
    if current is None or current != bound:
        raise ValueError("complete current source binding required")
    actual = tuple(
        ((m.start(), m.end()), m.group()) for m in re.finditer(r"\d+", parent)
    )
    if actual != tuple((span, literal) for span, literal, _, _ in _OWNERS):
        raise ValueError("complete numeric occurrence inventory differs")
    return tuple(InvestmentAdjustmentNumericOwner(*item) for item in _OWNERS)


def match_investment_adjustment_numeric_owner(
    bound, content, parent, guide, start, end, literal
):
    """Require an exact raw occurrence, never any enclosing clause or substring."""
    if type(start) is not int or type(end) is not int or type(literal) is not str:
        return None
    owners = investment_adjustment_numeric_owners(bound, content, parent, guide)
    matches = tuple(
        o for o in owners if o.span == (start, end) and o.literal == literal
    )
    return matches[0] if len(matches) == 1 else None


def match_investment_adjustment_typed_numeric_owner(
    bound, content, parent, guide, occurrence
):
    """Match exact extractor metadata; this still grants no execution authority."""
    from .validator_pipeline import NumericOccurrence

    if type(occurrence) is not NumericOccurrence:
        return None
    if (
        type(occurrence.start) is not int
        or type(occurrence.end) is not int
        or type(occurrence.raw) is not str
    ):
        return None
    owners = investment_adjustment_numeric_owners(bound, content, parent, guide)
    compound = (occurrence.start, occurrence.end, occurrence.raw) == (
        574,
        591,
        "multiplied by 50%",
    )
    owner = next(
        (
            o
            for o in owners
            if (o.span, o.literal)
            == ((occurrence.start, occurrence.end), occurrence.raw)
        ),
        None,
    )
    if compound:
        # One independently reviewed typed compound, not an enclosing-span rule.
        if parent[574:591] != "multiplied by 50%":
            return None
        owner = InvestmentAdjustmentNumericOwner(
            (574, 591),
            "multiplied by 50%",
            "substantive_half_rate",
            ("resource_recovery_rate", "line28actualnonzero"),
        )
    if owner is None:
        return None
    rate = owner.role == "substantive_half_rate"
    value = 0.5 if rate else float(owner.literal)
    source_value = 50.0 if rate else value
    if (
        type(occurrence.value) is not float
        or occurrence.value != value
        or type(occurrence.source_value) is not float
        or occurrence.source_value != source_value
        or occurrence.has_rate_context is not rate
        or occurrence.requires_rate_context is not rate
        or occurrence.has_temporal_context is not False
        or occurrence.has_structural_context is not False
        or occurrence.is_word_number is not False
        or occurrence.requires_month_context is not False
        or type(occurrence.alternative_values) is not tuple
        or occurrence.alternative_values != ()
    ):
        return None
    return owner
