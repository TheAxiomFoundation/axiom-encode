"""USDA jurisdiction tables: label shapes beyond the #2012 column label.

The FY2027 SNAP cost-of-living memo flattens three more table shapes that
numeric recall and the formula-branch check misread:

* a row label `48 States and D.C. $217 ...` under an `Area` caption, with
  sibling rows for the closed jurisdiction set (Alaska, Hawaii, Guam, Virgin
  Islands);
* a table title `... for 48 States and D.C.` directly before the `Household
  Size` caption and the column label that the #2012 rule masks;
* the header `Area 1-2 3 4 5 6+ 48 States`, where `6+` is the last range label,
  not addition.

The label's `48` leaves both gates (typed numeric recall and the positional
occurrence view of #2336); every amount stays demanded. All windows are a
few words of the public USDA memo or synthetic.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness import source_completeness as sc
from tests.test_evals_complete_source_occurrence_text import (
    _recall_values,
    _scalar_module,
    _typed_unnamed,
)

PAGE = "us/guidance/usda/fns/snap-fy2027-cola/page-6"

MASKED = (
    # (a) row label under `Area`, first cell an amount, two or more siblings.
    "Area Maximum Amount 48 States and D.C. $769 Alaska $1,229 Hawaii $1,036",
    "Area Reporting Requirement Threshold 48 States and D.C. $150 Alaska $150 "
    "Hawaii $150 Guam $150 Virgin Islands $150",
    "Household Size Household Size Area 1-2 3 4 5 6+ 48 States and D.C. $217 "
    "$217 $229 Alaska $371 $371 $371 Hawaii $306 $306 $306",
    "Area Household Size 1 -2 3 4 5 6+ 48 States and DC $198 $198 Alaska $338 "
    "$338 Guam $435 $435",
    # (b) table title before the masked column label.
    "Maximum SNAP Allotments for 48 States and D.C. Household Size 48 States "
    "and D.C. 1 $306 2 $562",
)

DEMANDED = (
    # Not a jurisdiction-set label.
    "Area Maximum Amount 48 months $769 Alaska $1,229 Hawaii $1,036",
    "Area Maximum Amount 48 hours $769 Alaska $1,229 Hawaii $1,036",
    "Area Maximum Amount $48 Alaska $1,229 Hawaii $1,036",
    "Area 48 contiguous States $769 Alaska $1,229 Hawaii $1,036",
    # A rule about the 48 States in running prose: no `Area` caption.
    "The 48 States and D.C. must pay $769. Alaska $1,229 Hawaii $1,036",
    # A lone row: one sibling only.
    "Area Maximum Amount 48 States and D.C. $769 Alaska $1,229",
    # No amount directly after the label.
    "Area Maximum Amount 48 States and D.C. pay 769. Alaska $1,229 Hawaii $1,036",
    # A currency sign between caption and label: not the first row.
    "Area Maximum Amount $5 48 States and D.C. $769 Alaska $1,229 Hawaii $1,036",
    # A title not followed by the masked column label.
    "Maximum SNAP Allotments for 48 States and D.C. Household Size 1 $306",
    "Maximum SNAP Allotments for 48 States and D.C. are $306 per person.",
)


@pytest.mark.parametrize("window", MASKED)
def test_label_48_leaves_both_gates(window):
    assert 48.0 not in _recall_values(window, PAGE)
    assert 48.0 not in _typed_unnamed(_scalar_module(PAGE, None), window, PAGE)


@pytest.mark.parametrize("window", DEMANDED)
def test_other_48_stays_demanded_by_both_gates(window):
    assert 48.0 in _recall_values(window, PAGE)
    assert 48.0 in _typed_unnamed(_scalar_module(PAGE, None), window, PAGE)


def test_amounts_of_a_masked_row_stay_demanded():
    window = MASKED[1]
    assert _recall_values(window, PAGE) == [150.0] * 5


@pytest.mark.parametrize(
    "source_text",
    (
        "Household Size Household Size Area 1-2 3 4 5 6+ 48 States and D.C. $217",
        "Area Household Size 1 -2 3 4 5 6+ 48 States and DC $198 $198",
        "Household Size 1 2 3 4 5 6+ 48 States & District of Columbia $209",
    ),
)
def test_range_header_plus_label_is_not_arithmetic(source_text):
    assert not sc._has_substantive_arithmetic_expression(source_text)


@pytest.mark.parametrize(
    "source_text",
    (
        "The amount is 6 + 48 dollars.",
        "Add 6+ 48 to the total.",
        "The amount equals income + 48.",
        "Area 1-2 3 equals 6 + 48 dollars.",
        # Words between the caption and `6+`: not a header row of range labels.
        "Household size 3 members add 6+ 48 dollars.",
    ),
)
def test_real_addition_stays_arithmetic(source_text):
    assert sc._has_substantive_arithmetic_expression(source_text)
