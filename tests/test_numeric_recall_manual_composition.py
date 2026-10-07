"""Citation masks must not turn operative US manual rows into section headings."""

from __future__ import annotations

import random

import pytest

from tests.test_numeric_recall_structural_properties import (
    PROFILES,
    _inventory,
    _recall_issues,
)

MANUAL = "us-il/manual/dhs/csmm/18929/block-4"


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("locator", ("WAC 388-450-0015", "Form FNS-380-1"))
def test_manual_locator_cleanup_still_precedes_heading_recognition(locator, profile):
    source = f"214.3 Telephone Allowance {locator}\nThe payment is $35."
    assert [item.value for item in _inventory(source, MANUAL, profile)] == [35]
    assert _recall_issues(source, MANUAL, profile), source
    assert not _recall_issues(source, MANUAL, profile, (35,)), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "source,values,missing_amount_values",
    (
        ("75.38 AABD Cash Payment (section 7C(1)(a))\n", (75.38,), ()),
        ("44.50 Countable Earned Income (section 7C(1)(a))\n", (44.5,), ()),
        ("75.38 AABD Cash Payment section 7C(1)(a)\n", (75.38,), ()),
        ("7.50 AABD Cash Payment (section 7.50C(1)(a))\n", (7.5,), ()),
        (
            "$470.00 Supplemental Security Income (SSI)\n"
            "44.50 Countable Earned Income (section 7C(1)(a))\n",
            (470, 44.5),
            (470,),
        ),
        (
            "44.50 Countable Earned Income (section 7C(1)(a))\n"
            "$470.00 Supplemental Security Income (SSI)\n",
            (44.5, 470),
            (470,),
        ),
    ),
)
def test_complete_citation_mask_keeps_manual_amount_row_recall(
    source, values, missing_amount_values, profile
):
    # Both gate directions matter: structural citation numbers add no recall
    # obligation, and omitting the independently operative row amount fails.
    assert [item.value for item in _inventory(source, MANUAL, profile)] == list(values)
    assert _recall_issues(source, MANUAL, profile, missing_amount_values), source
    assert not _recall_issues(source, MANUAL, profile, values), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("neighbor", ("", "before", "after"))
def test_generated_complete_citations_never_reclassify_manual_amount_rows(
    profile, neighbor
):
    """Renumbering or removing a citation preserves every operative amount."""

    generator = random.Random(17561779)
    amounts = (4438, 7538, *generator.sample(range(101, 99999), 6))
    for cents in amounts:
        amount = cents / 100
        written = f"{amount:.2f}"
        for target in ("7C(1)(a)", f"{written}C(1)(a)"):
            for citation in (f"(section {target})", f"section {target}."):
                source = f"{written} AABD Cash Payment {citation}\n"
                if neighbor == "before":
                    source = "$470.00 Supplemental Security Income (SSI)\n" + source
                elif neighbor == "after":
                    source += "$470.00 Supplemental Security Income (SSI)\n"
                values = (amount,) if not neighbor else (amount, 470)
                assert sorted(
                    item.value for item in _inventory(source, MANUAL, profile)
                ) == sorted(values), source
                assert _recall_issues(
                    source, MANUAL, profile, () if not neighbor else (470,)
                ), source
                assert not _recall_issues(source, MANUAL, profile, values), source
