"""Exact MHRtC numeric roles; static mapping, never execution authority."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from .bound_mhrtc_program import bind_mhrtc_program
from .mhrtc_program_contract import CITATION

# Independently reviewed raw-source occurrences, including repeated/overlapping
# printed labels. No case names, ordinals or caller-supplied coverage flags.
_RAW_ROLES = (
    (
        {
            "value": 0.145,
            "start": 2369,
            "end": 2373,
            "raw": "14.5",
            "has_rate_context": True,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 14.499999999999998,
            "requires_rate_context": True,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "credit_rate",
        (
            "federal_2025_mhrtc_rate",
            "multigenerational_home_renovation_tax_credit_line_45355",
        ),
    ),
    (
        {
            "value": 8.0,
            "start": 7532,
            "end": 7536,
            "raw": "×\n\n8",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 8.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "selected_total_qualifying_expenditures",
            "federal_2025_mhrtc_total_qualifying_expenditures_line_45354",
            "multigenerational_home_renovation_tax_credit_line_45355",
        ),
    ),
    (
        {
            "value": 1.0,
            "start": 7168,
            "end": 7169,
            "raw": "1",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 1.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "line_one",
            "line_two",
            "line_three",
            "selected_other_claim_amount",
            "selected_claim_capacity",
            "selected_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 2.0,
            "start": 7176,
            "end": 7177,
            "raw": "2",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 2.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "line_one",
            "line_two",
            "line_three",
            "selected_other_claim_amount",
            "selected_claim_capacity",
            "selected_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 10.0,
            "start": 7549,
            "end": 7551,
            "raw": "10",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 10.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "selected_total_qualifying_expenditures",
            "federal_2025_mhrtc_total_qualifying_expenditures_line_45354",
            "multigenerational_home_renovation_tax_credit_line_45355",
        ),
    ),
    (
        {
            "value": 2025.0,
            "start": 70,
            "end": 74,
            "raw": "2025",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 2025.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "closed_2025_version",
        (),
    ),
    (
        {
            "value": 12.0,
            "start": 940,
            "end": 942,
            "raw": "12",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 12.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "claimant_occupancy_fact",
        ("eligible_claimant",),
    ),
    (
        {
            "value": 50000.0,
            "start": 1770,
            "end": 1776,
            "raw": "50,000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 50000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "shared_cap",
        (
            "selected_eligible_other_claims",
            "selected_other_claim_amount",
            "selected_claim_capacity",
        ),
    ),
    (
        {
            "value": 50000.0,
            "start": 1931,
            "end": 1937,
            "raw": "50,000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 50000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "shared_cap",
        (
            "selected_eligible_other_claims",
            "selected_other_claim_amount",
            "selected_claim_capacity",
        ),
    ),
    (
        {
            "value": 50000.0,
            "start": 2270,
            "end": 2276,
            "raw": "50,000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 50000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "per_renovation_cap",
        (
            "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation",
            "selected_claim_capacity",
            "selected_total_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 50000.0,
            "start": 2407,
            "end": 2413,
            "raw": "50,000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 50000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "per_renovation_cap",
        (
            "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation",
            "selected_claim_capacity",
            "selected_total_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 7250.0,
            "start": 2498,
            "end": 2503,
            "raw": "7,250",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 7250.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "per_renovation_credit_max",
        (
            "federal_2025_mhrtc_maximum_credit_per_renovation",
            "multigenerational_home_renovation_tax_credit_line_45355",
        ),
    ),
    (
        {
            "value": 5000.0,
            "start": 2916,
            "end": 2920,
            "raw": "5000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 5000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "form_identifier",
        (),
    ),
    (
        {
            "value": 12.0,
            "start": 3209,
            "end": 3211,
            "raw": "12",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 12.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "dwelling_occupancy_facts",
        ("eligible_dwelling",),
    ),
    (
        {
            "value": 1.24,
            "start": 3618,
            "end": 3622,
            "raw": "1.24",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 1.24,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "source_equivalent_display_unit",
        "land_general_scope",
        (
            "general_land_limit_hectares",
            "land_extent_eligible",
            "receipt_basic_conditions",
        ),
    ),
    (
        {
            "value": 65.0,
            "start": 4212,
            "end": 4214,
            "raw": "65",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 65.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "individual_age65",
        ("qualifying_individual",),
    ),
    (
        {
            "value": 18.0,
            "start": 4239,
            "end": 4241,
            "raw": "18",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 18.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "individual_age18_dtc",
        ("qualifying_individual",),
    ),
    (
        {
            "value": 18.0,
            "start": 4350,
            "end": 4352,
            "raw": "18",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 18.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "relation_age18",
        ("qualifying_relation",),
    ),
    (
        {
            "value": 2026.0,
            "start": 5658,
            "end": 5662,
            "raw": "2026",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 2026.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "illustrative_date_role",
        "completion_year",
        ("completion_in_tax_year", "claim_source_eligibility"),
    ),
    (
        {
            "value": 2025.0,
            "start": 5695,
            "end": 5699,
            "raw": "2025",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 2025.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "illustrative_date_role",
        "completion_year",
        ("completion_in_tax_year", "claim_source_eligibility"),
    ),
    (
        {
            "value": 5000.0,
            "start": 6000,
            "end": 6004,
            "raw": "5000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 5000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "form_identifier",
        (),
    ),
    (
        {
            "value": 50000.0,
            "start": 6925,
            "end": 6931,
            "raw": "50,000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 50000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "per_renovation_cap",
        (
            "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation",
            "selected_claim_capacity",
            "selected_total_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 50000.0,
            "start": 7149,
            "end": 7158,
            "raw": "50,000.00",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 50000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "per_renovation_cap",
        (
            "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation",
            "selected_claim_capacity",
            "selected_total_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 3.0,
            "start": 7178,
            "end": 7179,
            "raw": "3",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 3.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "line_one",
            "line_two",
            "line_three",
            "selected_other_claim_amount",
            "selected_claim_capacity",
            "selected_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 4.0,
            "start": 7181,
            "end": 7182,
            "raw": "4",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 4.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "line_one",
            "line_two",
            "line_three",
            "selected_other_claim_amount",
            "selected_claim_capacity",
            "selected_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 5.0,
            "start": 7183,
            "end": 7184,
            "raw": "5",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 5.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "line_one",
            "line_two",
            "line_three",
            "selected_other_claim_amount",
            "selected_claim_capacity",
            "selected_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 6.0,
            "start": 7185,
            "end": 7186,
            "raw": "6",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 6.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "line_one",
            "line_two",
            "line_three",
            "selected_other_claim_amount",
            "selected_claim_capacity",
            "selected_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 7.0,
            "start": 7248,
            "end": 7249,
            "raw": "7",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 7.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "line_one",
            "line_two",
            "line_three",
            "selected_other_claim_amount",
            "selected_claim_capacity",
            "selected_qualifying_expenditures",
        ),
    ),
    (
        {
            "value": 45354.0,
            "start": 7412,
            "end": 7417,
            "raw": "45354",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 45354.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "public_total",
        (
            "selected_total_qualifying_expenditures",
            "federal_2025_mhrtc_total_qualifying_expenditures_line_45354",
            "multigenerational_home_renovation_tax_credit_line_45355",
        ),
    ),
    (
        {
            "value": 8.0,
            "start": 7535,
            "end": 7536,
            "raw": "8",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 8.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "printed_line_locator",
        (
            "selected_total_qualifying_expenditures",
            "federal_2025_mhrtc_total_qualifying_expenditures_line_45354",
            "multigenerational_home_renovation_tax_credit_line_45355",
        ),
    ),
    (
        {
            "value": 7250.0,
            "start": 7608,
            "end": 7613,
            "raw": "7,250",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 7250.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "executable_semantic_role",
        "per_renovation_credit_max",
        (
            "federal_2025_mhrtc_maximum_credit_per_renovation",
            "multigenerational_home_renovation_tax_credit_line_45355",
        ),
    ),
    (
        {
            "value": 5000.0,
            "start": 7682,
            "end": 7686,
            "raw": "5000",
            "has_rate_context": False,
            "has_temporal_context": False,
            "has_structural_context": False,
            "source_value": 5000.0,
            "requires_rate_context": False,
            "is_word_number": False,
            "alternative_values": (),
            "requires_month_context": False,
        },
        "structural_source_role",
        "form_identifier",
        (),
    ),
)


def _typed(value):
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("string field names required")
        return (dict, tuple(sorted((k, _typed(v)) for k, v in value.items())))
    if type(value) is tuple:
        return (tuple, tuple(_typed(v) for v in value))
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("finite numeric occurrence required")
        return (float, value.hex())
    if value is None or type(value) in (bool, int, str):
        return (type(value), value)
    raise ValueError("unsupported occurrence field type")


@dataclass(frozen=True)
class MhrtcNumericRole:
    recall_span: tuple[int, int]
    source_span: tuple[int, int]
    literal: str
    classification: str
    role: str
    rules: tuple[str, ...]


def mhrtc_numeric_roles(bound, content, source, occurrences):
    """Verify complete production inventory and its exact source projection.

    Returned roles are static support only. A separate private execution owner
    must prove the substantive witnesses before granting any coverage credit.
    """
    from . import source_completeness as sc
    from .validator_pipeline import (
        NumericOccurrence,
        extract_typed_numeric_inventory_occurrences_from_text,
    )

    current = bind_mhrtc_program(content, source)
    if current is None or current != bound:
        raise ValueError("complete current source/formula binding required")
    if type(occurrences) not in (list, tuple):
        raise ValueError("complete occurrence sequence required")
    if any(type(o) is not NumericOccurrence for o in occurrences):
        raise ValueError("actual typed numeric occurrences required")
    # M's binder requires the exact full source; production uses stripped text.
    stripped = source.strip()
    leading = len(source) - len(source.lstrip())
    source_map = sc._NumericRecallSourceMap(list(range(len(stripped))))
    recall = sc.authoritative_numeric_recall_text(
        stripped, corpus_citation_path=CITATION, _source_map=source_map
    )
    actual = tuple(
        o
        for o in extract_typed_numeric_inventory_occurrences_from_text(
            recall, profile="legacy"
        )
        if not sc._temporal_occurrence_is_formula_applicability_preface(o, recall)
    )
    if Counter(_typed(asdict(o)) for o in occurrences) != Counter(
        _typed(asdict(o)) for o in actual
    ):
        raise ValueError("complete production numeric inventory differs")
    expected = {
        _typed(row): (classification, role, rules)
        for row, classification, role, rules in _RAW_ROLES
    }
    if len(expected) != len(_RAW_ROLES):
        raise ValueError("ambiguous static occurrence identity")
    projected = []
    result = []
    for occurrence in actual:
        span = source_map.source_span(occurrence.start, occurrence.end)
        if span is None:
            raise ValueError("noncontiguous numeric source origin")
        raw_span = (span[0] + leading, span[1] + leading)
        if source[raw_span[0] : raw_span[1]] != occurrence.raw:
            raise ValueError("numeric source substring differs")
        row = asdict(occurrence)
        row.update(start=raw_span[0], end=raw_span[1])
        key = _typed(row)
        if key not in expected:
            raise ValueError("unowned numeric source occurrence")
        classification, role, rules = expected[key]
        projected.append(key)
        result.append(
            MhrtcNumericRole(
                (occurrence.start, occurrence.end),
                raw_span,
                occurrence.raw,
                classification,
                role,
                rules,
            )
        )
    if Counter(projected) != Counter(_typed(row) for row, *_ in _RAW_ROLES):
        raise ValueError("complete raw numeric occurrence multiset differs")
    return tuple(result)
