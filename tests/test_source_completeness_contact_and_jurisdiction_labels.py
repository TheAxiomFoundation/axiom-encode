"""Addresses and jurisdiction-set table labels are not numeric recall demands.

Issue #2003: a local encode of the USDA FNS SNAP FY 2026 income eligibility
standards (`fns_snap_fy2026_income_standards_composed.txt`, the exact
document-level source body) encoded the column label `48 States, District of
Columbia, Guam, Virgin Islands` as `standard_table_state_count = 48`, and the
FNS mailing address `1320 Braddock Place, Alexandria, VA 22314` as
`fns_street_address_number = 1320`. Each mask here removes only those
identifier numbers; every amount, including one beside the label or the
address, stays demanded.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

SNAP_DOCUMENT = "us/guidance/usda/fns/snap-fy2026-income-eligibility-standards"
GENERIC = "us/guidance/example/page-1"
COMPOSED = (
    Path(__file__).parent
    / "fixtures/source_completeness/fns_snap_fy2026_income_standards_composed.txt"
)


def _inventory(source: str, citation: str) -> list[tuple[float, int]]:
    recall = sc.authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    return [
        (float(item.value), item.start)
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            recall, profile="legacy"
        )
    ]


def _values(source: str, citation: str = GENERIC) -> list[float]:
    return [value for value, _ in _inventory(source, citation)]


def _demanded(source: str, citation: str, scalars=()) -> set[float]:
    candidate = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {},
            "rules": [
                {"name": f"supplied_{index}", "versions": [{"formula": value}]}
                for index, value in enumerate(scalars)
            ],
        }
    )
    issues = sc.analyze_complete_source_unit(
        candidate,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues
    prefix = "[complete-source-unit:numeric-recall] Authoritative corpus numeric value "
    return {
        float(issue[len(prefix) :].split(" ", 1)[0])
        for issue in issues
        if issue.startswith(prefix)
    }


# --- The #2003 source ---------------------------------------------------------


def test_snap_fy2026_document_demands_no_label_or_address_number():
    source = COMPOSED.read_text().strip()
    assert source.count("Household Size 48 States, District of Columbia") == 3
    assert "1320 Braddock Place, Alexandria, VA 22314" in source

    values = _values(source, SNAP_DOCUMENT)
    assert 48 not in values
    assert 1320 not in values
    assert 22314 not in values

    # Only the three label occurrences leave the inventory. A same-length
    # spelling that the recognizer rejects (`Stat3s`) keeps them, and every
    # other occurrence keeps its value and offset.
    unmasked = _inventory(source.replace("48 States", "48 Stat3s"), SNAP_DOCUMENT)
    labels = [item for item in unmasked if item[0] == 48]
    assert len(labels) == 3
    assert _inventory(source, SNAP_DOCUMENT) == [
        item for item in unmasked if item[0] != 48
    ]

    demanded = _demanded(source, SNAP_DOCUMENT)
    assert not demanded & {48, 1320, 22314}
    # Every published table amount stays demanded.
    assert {1305, 1630, 1500, 459, 574, 528, 1696, 596, 2152, 8567, 871} <= demanded


def test_snap_fy2026_page_paths_keep_their_existing_label_mask():
    source = COMPOSED.read_text().strip()
    for citation in (f"{SNAP_DOCUMENT}/page-1", f"{SNAP_DOCUMENT}/page-2"):
        assert 48 not in _values(source, citation)
        assert 1305 in _values(source, citation)


# --- Jurisdiction-set column labels ------------------------------------------


@pytest.mark.parametrize(
    "label",
    (
        "Household Size 48 States, District of Columbia, Guam, Virgin Islands",
        "Household Size 48 States, D.C., Guam, Virgin Islands",
        "Household Size 48 States, DC, Guam, Virgin Islands",
        "Household Size 48 States & District of Columbia",
        "Household Size 48 States and the District of Columbia",
        "Household size 48 States, Guam, Virgin Islands",
        "Household Size\n48 States, District of Columbia",
    ),
)
def test_jurisdiction_set_column_label_is_not_demanded(label):
    source = f"{label} Alaska Hawaii 1 $1,305 $1,630 $1,500 2 $1,763"
    values = _values(source)
    assert 48 not in values
    assert {1, 2, 1305, 1630, 1500, 1763} <= set(values)


@pytest.mark.parametrize(
    "source",
    (
        # Not a jurisdiction-set label.
        "Household Size 48 months, District of Columbia",
        "The certification period is 48 months.",
        "Household Size 48 States receive the standard.",
        "Household Size 48 states, District of Columbia",
        "Household Size 48 Statesmen, DC",
        "Household Size 48 States, Districts",
        "Household Size 148 States, DC",
        "Household Size 48.5 States, DC",
        # A currency, sign, number or other word before it.
        "The amount is $48 States, District of Columbia",
        "The amount is $ 48 States, District of Columbia",
        "The amount is USD 48 States and DC",
        "At least 48 States and DC must report.",
        "More than 48 States, District of Columbia",
        "Household Size 1 2 3 4 5 6+ 48 States & District of Columbia",
        "Household Size 8 48 States, DC",
        "Household Size -48 States, DC",
        "Each of 48 States, District of Columbia",
        "A heading\n48 States, District of Columbia",
        # Prose, even definite, can state an operative count (a quorum).
        "48 States and DC must approve the compact before it takes effect.",
        "The 48 States and DC must approve the compact.",
        "48 States & Virgin Islands",
        # A table row, not a header: no first household-size row follows.
        "Criterion\tLimit\tApplies in\nMaximum Household Size\t48\tStates, "
        "District of Columbia and Guam",
        "Maximum Household Size 48 States, District of Columbia and Guam",
        "Maximum  Household Size 48 States, District of Columbia and Guam",
        "Maximum\tHousehold Size 48 States, District of Columbia and Guam",
        "Maximum\nHousehold Size 48 States, District of Columbia and Guam",
        "Maximum: Household Size 48 States, District of Columbia and Guam",
        "Household Size 48  States, District of Columbia",
    ),
)
@pytest.mark.parametrize(
    "rows", ("", " Alaska Hawaii 1 $1,305 $1,630 $1,500"), ids=["bare", "rows"]
)
def test_48_stays_demanded_outside_a_jurisdiction_set_label(source, rows):
    source += rows
    assert list(sc._contiguous_states_numeric_recall_spans(source)) == []
    number = 148 if "148" in source else 48.5 if "48.5" in source else 48
    assert number in {abs(value) for value in _values(source)}


@pytest.mark.parametrize(
    "after",
    (
        " Alaska Hawaii",
        " Alaska Hawaii 2 $1,763",
        " Alaska Hawaii 1 1,305",
        " Alaska Hawaii are listed in Table 2. 1 $1,305",
        " " + "Alaska " * 40 + "1 $1,305",
        "\nHousehold Size Monthly Limit\n1 $1,305",
    ),
)
def test_label_without_a_first_household_size_row_stays_demanded(after):
    source = "Household Size 48 States, District of Columbia, Guam" + after
    assert 48 in _values(source)


def test_dollar_amount_in_a_48_states_column_stays_demanded():
    source = (
        "Household Size 48 States, District of Columbia, Guam, Virgin Islands "
        "Alaska Hawaii 1 $48 $60 $55 Each Additional Member $48 $574 $528"
    )
    recall = sc.authoritative_numeric_recall_text(source, corpus_citation_path=GENERIC)
    assert recall.count("$48") == 2
    assert _values(source).count(48) == 2
    assert 48 in _demanded(source, GENERIC)
    assert 48 not in _demanded(source, GENERIC, scalars=(48, 60, 55, 574, 528, 1))


def test_count_the_rule_uses_stays_demanded():
    source = (
        "Household Size 48 States, District of Columbia, Guam, Virgin Islands "
        "Alaska Hawaii 1 $1,305. A State agency may certify a household for up "
        "to 48 months."
    )
    inventory = _values(source)
    assert inventory.count(48) == 1
    assert 48 in _demanded(source, GENERIC)


# --- Postal addresses -----------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "numbers"),
    (
        ("1320 Braddock Place, Alexandria, VA 22314", (1320, 22314)),
        ("1320 Braddock Place, Alexandria, VA 22314-1500", (1320, 22314, 1500)),
        # A ZIP+4 extension cut off at a page or line end.
        ("3388 Brentwood Drive, Baton Rouge, LA 70809-", (3388, 70809)),
        ("1400 Independence Ave., SW, Washington, DC 20250", (1400, 20250)),
        ("800 I Street, NW, Washington, DC 20001 - 3710", (800, 20001, 3710)),
        ("1400 Independence Avenue SW, Washington, DC 20250", (1400, 20250)),
        ("61 Forsyth St. SW, Suite 8T36, Atlanta, GA 30303", (61, 30303)),
        ("90 Seventh Street, Suite 10-100, San Francisco, CA 94103", (90, 94103)),
        ("90 Seventh Street, Suite 10/100, San Francisco, CA 94103", None),
    ),
)
def test_complete_postal_address_numbers_are_not_demanded(address, numbers):
    source = f"Food and Nutrition Service, {address}. The limit is $1,305."
    values = _values(source)
    assert 1305 in values
    if numbers is None:
        # `Suite 10/100` is not a recognized unit: the address is kept whole
        # (fails toward demanding).
        recall = sc.authoritative_numeric_recall_text(
            source, corpus_citation_path=GENERIC
        )
        assert address in recall
        assert 90 in values
        return
    assert not set(numbers) & set(values)


@pytest.mark.parametrize(
    "source",
    (
        # An amount, not a house number.
        "The deduction is $1,320 Braddock Place, Alexandria, VA 22314.",
        "The deduction is $1320 Braddock Place, Alexandria, VA 22314.",
        "The deduction is $ 1320 Braddock Place, Alexandria, VA 22314.",
        "The deduction is USD 1320 Braddock Place, Alexandria, VA 22314.",
        "The deduction is USD  1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is USD\u00a01320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is $\u202f1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is $\n1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is USD" + " " * 14 + "1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is $" + " " * 16 + "1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is USD -1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is $-1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is USD - 1320 Braddock Pl., Alexandria, VA 22314.",
        "The deduction is $ \u2212 1320 Braddock Pl., Alexandria, VA 22314.",
        "The change is -1320 Braddock Pl., Alexandria, VA 22314.",
        "The rate is 0.1320 Braddock Place, Alexandria, VA 22314.",
        # No complete city, state and ZIP tail.
        "Send it to 1320 Braddock Place.",
        "Send it to 1320 Braddock Place, Alexandria, Virginia.",
        "Send it to 1320 Braddock Place, Alexandria, XX 22314.",
        "Send it to 1320 Braddock Place, Alexandria, VA 223145.",
        # No street suffix.
        "Households with 1320 Eligible Members, Alexandria, VA 22314.",
    ),
)
def test_number_outside_a_complete_postal_address_stays_demanded(source):
    values = {abs(value) for value in _values(source)}
    assert any(value in values for value in (1320, 0.132))


@pytest.mark.parametrize(
    "tail",
    (
        "\n- 1500 dollars is the maximum monthly income.",
        " - 1500 dollars is the maximum monthly income.",
        "\n-1500 dollars is the maximum monthly income.",
        "\n- 1500, the maximum monthly income.",
        "\n- 1500\n  dollars is the maximum monthly income.",
        "\n- 1500.",
    ),
)
def test_amount_after_an_address_is_not_read_as_a_zip_extension(tail):
    source = f"Mail applications to 1320 Braddock Place, Alexandria, VA 22314{tail}"
    assert 1500 in {abs(value) for value in _values(source)}
    assert 1320 not in _values(source)


@pytest.mark.parametrize(
    "prefix", ("Mail applications to", "The deduction is USD", "The deduction is $")
)
def test_bullet_after_an_address_or_kept_amount_stays_demanded(prefix):
    source = (
        f"{prefix} 1320 Braddock Place, Alexandria, VA 22314\n"
        "- 1500 dollars is the maximum monthly income."
    )
    assert 1500 in {abs(value) for value in _values(source)}


def test_split_zip_plus_four_is_still_masked():
    for source in (
        "Write to Alexandria, VA 22314 - 1500.",
        "Write to Alexandria, VA 22314-\n1500\nPhone",
        "Write to 1320 Braddock Place, Alexandria, VA 22314-\n1500 Phone",
        "Write to 800 I Street, NW, Washington, DC 20001 - 3710; phone",
        # Virginia Form 760 instructions: a locality contact list.
        "Charlottesville City - 540 P.O. Box 2964, Charlottesville, VA 22902 -2964 "
        "434-970-3160 Chesapeake City",
    ):
        values = _values(source)
        assert not {1500, 3710, 2964} & {abs(value) for value in values}, source


def test_amount_on_the_same_line_as_an_address_stays_demanded():
    source = (
        "Food and Nutrition Service, 1320 Braddock Place, Alexandria, VA 22314. "
        "The maximum allotment is $1,320 and the shelter cap is $22,314."
    )
    values = _values(source)
    assert values.count(1320) == 1
    assert values.count(22314) == 1
    assert {1320, 22314} <= _demanded(source, GENERIC)
