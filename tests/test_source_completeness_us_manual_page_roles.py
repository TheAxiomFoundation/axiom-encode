"""Page furniture on a sectioned US manual page is not a rule value.

State SNAP manual pages were refused on numeric recall, and on the encode
eval's occurrence count, for numbers that only organize the page: the running
header (page, section, change notice), the document's own section labels,
and list markers flattened into a line. The numbers of a worked example are
not a recognized shape: they stay demanded.

Each rule is a positive shape on a `.../<prefix>-<NNN>-<slug>/page-<P>` manual
path. A number next to a similar shape keeps its demand. Both gates read the
same view: typed numeric recall and the positional occurrence view.

All source text here is synthetic.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_numeric_recall_see_example_reference import KEPT as SEE_EXAMPLE_KEPT

PAGE = "us-zz/manual/agency/fns/fns-315-special-budgeting/page-16"
OTHER_SECTION_PAGE = "us-zz/manual/agency/fns/fns-340-deductions/page-16"
SECTION_130_PAGE = "us-zz/manual/agency/fns/fns-130-income-limits/page-16"
UNSECTIONED_MANUAL = "us-zz/manual/agency/handbook/shelter-costs/block-3"
REGULATION = "us-zz/regulation/fns-315-special-budgeting/page-16"
PROFILES = ("legacy", "en-US")
GLYPH_SECTION = "SecƟon"
TITLE = "AGENCY SERVICES CERTIFICATION INCOME FNS 315 SPECIAL BUDGETING\n\n"
HEADER = TITLE + "16 Section 315 Change #01-2022 January 11, 2022 "
PREFIX = "[complete-source-unit:numeric-recall] Authoritative corpus numeric value "


def _extractor(profile: str):
    return functools.partial(
        extract_typed_numeric_inventory_occurrences_from_text, profile=profile
    )


def _occurrence_view(source: str, citation: str, profile: str) -> list[float]:
    """The numbers the encode eval's occurrence count demands."""

    return sorted(
        occurrence.value
        for occurrence in sc.complete_source_numeric_recall_occurrences(
            source,
            corpus_citation_path=citation,
            extract_numeric_occurrences=_extractor(profile),
        )
    )


def _typed_recall(
    source: str, citation: str, profile: str, scalars: tuple[float, ...] = ()
) -> list[float]:
    """The values typed numeric recall demands of a module with `scalars`."""

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
        extract_numeric_occurrences=_extractor(profile),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues
    return sorted(
        {
            float(issue[len(PREFIX) :].split(" ", 1)[0])
            for issue in issues
            if issue.startswith(PREFIX)
        }
    )


def _assert_both_gates(source: str, citation: str, profile: str, demanded) -> None:
    view = _occurrence_view(source, citation, profile)
    assert sorted(set(view)) == sorted(set(demanded)), view
    assert _typed_recall(source, citation, profile) == sorted(set(demanded))


def _assert_both_gates_keep(source: str, citation: str, profile: str, kept) -> None:
    """Both gates demand the same values, and `kept` is among them.

    The two numeric profiles read some neighbors differently (`#01-2022`,
    `PERCENT`), so only the values under test are named. An empty `kept`
    means that some value must stay demanded.
    """

    view = set(_occurrence_view(source, citation, profile))
    assert view and set(kept) <= view, view
    assert _typed_recall(source, citation, profile) == sorted(view)


# (source, values still demanded)
ACCEPTED = (
    # The running header: page, section, change notice. The date stays a date.
    (HEADER + "Count the payment of $40 in full.", [40.0]),
    (
        TITLE.rstrip() + " 16 Section 315 Change #01-2022 January 11, 2022 "
        "Count the payment of $40 in full.",
        [40.0],
    ),
    (
        TITLE + f"16 {GLYPH_SECTION} 315 Change #3 2017 September 20, 2017 "
        "Count the payment of $40 in full.",
        [40.0],
    ),
    (
        TITLE + "16 Section 315 Change #6-20XX July 12, 2021 "
        "Count the payment of $40 in full.",
        [40.0],
    ),
    # Header variants: spaced dash, no `#`, no section part, `FNS NNN`, and a
    # page number on its own line.
    (
        TITLE + "16 Section 315 Change #3 -2017 September 20, 2017 "
        "Count the payment of $40 in full.",
        [40.0],
    ),
    (
        TITLE + "16 Section 315 Change 03-2021 June 1, 2021 Count $40 in full.",
        [40.0],
    ),
    (TITLE + "16 Change #1-2021 May 1, 2021 Count $40 in full.", [40.0]),
    (TITLE + "16 FNS 315 Change #6-2021 July 12, 2021 Count $40 in full.", [40.0]),
    (
        TITLE + "16\n\nSection 315 Change #3-2017 September 20, 2017 "
        "Count the payment of $40 in full.",
        [40.0],
    ),
    # A heading of the document's own section: after a sentence end, at a
    # line start, or directly after the page header.
    (
        "Count the payment of $40 in full. 315.27 REPRESENTATIVE PAYMENTS "
        "A payee receives the money.",
        [40.0],
    ),
    (
        "The payment of $40 is counted in full.\n\n315.30 STRIKERS AND "
        "LOCKOUTS A. Count it.",
        [40.0],
    ),
    ("Count $40. 315.06 COLLECTING THE AE CLAIM\n\nA. Start it.", [40.0]),
    (HEADER + "315.03 INCOME STANDARDS The limit is $40.", [40.0]),
    # A reference to the document's own section.
    ("Count $40 in full. If there was a loss, see 315.28 F. Enter it.", [40.0]),
    ("Count $40 in full. See 315.28 C. for the list.", [40.0]),
    ("Count $40 in full, as shown in 315.02, A. Enter it.", [40.0]),
    ("Count $40 in full. See 315.06, A, 4.\n\nEnter the claim.", [40.0]),
    ("Count $40 in full (see 315.28).", [40.0]),
    # The glyph spelling of `Section` reads as `Section`.
    (f"Refer to {GLYPH_SECTION} 210, Household Concept, for the $40 limit.", [40.0]),
    (f"A person defined in {GLYPH_SECTION} 212.02, Person, has $40.", [40.0]),
    # List markers that open a sentence and belong to a run.
    (
        "Give the unit time to verify. 2. If the unit paid $40, allow it. "
        "3. Do not ask for more.",
        [40.0],
    ),
    (
        "Verify the cost; 1. Ask the unit; 2. Ask the provider; 3. Allow $40.",
        [40.0],
    ),
    ("1. Give the unit time.\n2. Allow $40. 3. Do not ask for more.", [40.0]),
    # The first marker of the page body, with its successor on the page.
    (HEADER + "4. A date must be set. 5. The claim is $40.", [40.0]),
)

# A number next to a similar shape keeps its demand.
STILL_DEMANDED = (
    # A header whose page or section is not the path's is not this header.
    (
        TITLE + "17 Section 315 Change #01-2022 January 11, 2022 Count $40.",
        [17.0, 2022.0, 40.0],
    ),
    (
        TITLE + "16 Section 999 Change #01-2022 January 11, 2022 Count $40.",
        [16.0, 2022.0, 40.0],
    ),
    # A change notice without its full date is not the header grammar.
    (TITLE + "16 Change #1-2021 applies to $40.", [16.0, 2021.0, 40.0]),
    (TITLE + "Pay $16 Change #1-2021 May 1, 2021 and $40.", [16.0, 40.0]),
    # No change notice: not the header grammar.
    ("16 Section 315 applies to $40.", [16.0, 40.0]),
    ("Pay 16 Section 315 Change #01-2022 dollars and $40.", [16.0, 2022.0, 40.0]),
    # An amount or rate that equals a section label.
    ("Count $315.27 REPRESENTATIVE PAYMENTS in full.", [315.27]),
    ("The rate is 315.27 PERCENT OF income.", []),
    ("The rate is 315.27% OF income.", []),
    ("Multiply by 315.27 and count it.", [315.27]),
    ("The total is $40 + 315.27 TOTAL DUE.", [40.0, 315.27]),
    ("The result is 315.275 TOTAL DUE.", [315.275]),
    ("Count 315.27 Representative payments.", [315.27]),
    # A label needs a sentence end, a line start or the header before it, and
    # a title word of four or more capital letters after it.
    ("The limit is 315.00 TANF and 200.00 SSI.", [315.0, 200.0]),
    ("The unit keeps 315.50 OF the grant.", [315.5]),
    ("The grant is 315.00 USD each month.", [315.0]),
    ("Multiply by 315.25 AND add the rest.", [315.25]),
    ("The household keeps 315.27 TOTAL each month.", [315.27]),
    ("Count it. 315.27 FPL applies.", [315.27]),
    ("Count it. 315.00 TANF applies.", [315.0]),
    # A title needs two words of four or more capital letters.
    ("Count it. 315.50 GROSS income is the limit.", [315.5]),
    ("Count it.\n315.50 TANF", [315.5]),
    ("Count it.\n\n315.30 STRIKERS A. Count the income.", [315.3]),
    ("Count it. 315.50 GROSS AND NET income.", [315.5]),
    # A label has exactly two decimals.
    ("Count $40 in full; see 315.5; then stop.", [40.0, 315.5]),
    ("Count $40 in full. See 315.6, A, 4.\n\nEnter it.", [40.0, 315.6, 4.0]),
    ("Count $40 in full, as shown in 315.2, A. Enter it.", [40.0, 315.2]),
    ("Count $40. 315.275 REPRESENTATIVE PAYMENTS apply.", [40.0, 315.275]),
    # A row under a quantity caption stays a value.
    ("Benefit amounts, in dollars:\n315.27 REPRESENTATIVE PAYMENTS", [315.27]),
    # Another document's section number is not this document's label.
    ("Count $40. 340.15 SPECIAL PROCEDURES apply.", [40.0, 340.15]),
    ("Count $40. See 340.15 C. for the list.", [40.0, 340.15]),
    # A reference needs `see`, or `in` with the comma and subdivision letter.
    ("The unit pays 315.28 F. Enter it.", [315.28]),
    ("See 315.28 dollars of income.", [315.28]),
    ("Count the $40 in 315.28 cases.", [40.0, 315.28]),
    ("The sum results in 315.50.", [315.5]),
    ("Count the rest in 315.00; then stop.", [315.0]),
    ("Count the rest (in 315.5) and stop.", [315.5]),
    ("The sum results in 315.50 F. Enter it.", [315.5]),
    # The glyph rule reads only what `Section` would.
    (f"The {GLYPH_SECTION} pays $210, and $40 more.", [210.0, 40.0]),
    # A number that ends a sentence is not a list marker.
    ("The limit is a household of 2. If more, deny.", [2.0]),
    ("Count income for children under 6. Advertising costs count.", [6.0]),
    ("Use Form No. 5. Complete it.", [5.0]),
    ("The limit is $2. If more, deny.", [2.0]),
    ("Multiply by 2.15 and then stop. The result is 3.5. Enter it.", [2.15, 3.5]),
    ("Wait. 2. 3 months must pass.", [2.0, 3.0]),
    # A lone number that opens a sentence belongs to no run.
    ("The period is approx. 30. Days are counted.", [30.0]),
    ("The age limit is (max.) 60. Elderly members qualify.", [60.0]),
    ("Use a short period, e.g. 10. The unit reports.", [10.0]),
    ("The unit states the age. 18. Students are exempt.", [18.0]),
    ("Count the share. 99. Percent applies.", []),
    ("Wait. 30. Days must pass. 31. Days is too long.", [30.0, 31.0]),
    ("Give the unit time. 2. If the unit paid $40, allow it.", [2.0, 40.0]),
    ("Use e.g. 1. Ask the unit. 2. Ask the provider.", [1.0]),
    ("See p. 4. Ask the unit. 5. Ask the provider.", [4.0]),
    ("See pg. 4. Ask the unit. 5. Ask the provider.", [4.0]),
    ("See Ch. 4. Ask the unit. 5. Ask the provider.", [4.0]),
    ("See par. 4. Ask the unit. 5. Ask the provider.", [4.0]),
    # The first marker of the page body needs a run too.
    (HEADER + "6. Advertising costs count.", [6.0]),
    (HEADER + "18. Students are exempt. The claim is $40.", [18.0, 40.0]),
    (HEADER + "4. A date must be set. The claim is $40.", [4.0, 40.0]),
    # Unit labels are not a recognized shape.
    ("FNS unit 1 has the child. FNS unit 2 has the parent.", [1.0, 2.0]),
    ("FNS unit 1: $291 FNS unit 2: $535", [1.0, 291.0, 2.0, 535.0]),
)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(("source", "demanded"), ACCEPTED)
def test_page_furniture_is_not_demanded_by_either_gate(source, demanded, profile):
    _assert_both_gates(source, PAGE, profile, demanded)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(("source", "demanded"), STILL_DEMANDED)
def test_similar_shape_keeps_its_demand_in_both_gates(source, demanded, profile):
    _assert_both_gates_keep(source, PAGE, profile, demanded)


@pytest.mark.parametrize("profile", PROFILES)
def test_amount_that_equals_the_own_section_keeps_its_demand(profile):
    _assert_both_gates_keep(
        "The limit is 130.00 FPL for the unit.", SECTION_130_PAGE, profile, [130.0]
    )
    for source, value in (
        ("Count it. 130.50 FPL applies.", 130.5),
        ("Count it. 130.50 GROSS income is the limit.", 130.5),
        ("Count it.\n130.50 TANF", 130.5),
        ("Count it; see 130.5; then stop.", 130.5),
    ):
        _assert_both_gates_keep(source, SECTION_130_PAGE, profile, [value])


@pytest.mark.parametrize(
    "citation", (UNSECTIONED_MANUAL, REGULATION, "us-zz/manual/agency/fns/page-16")
)
@pytest.mark.parametrize(("source", "_demanded"), ACCEPTED)
def test_rules_apply_only_on_a_sectioned_manual_page(source, _demanded, citation):
    assert sc._us_manual_page_role_spans(source, corpus_citation_path=citation) == ()


def test_own_section_comes_from_the_citation_path():
    source = "Count $40. 315.27 REPRESENTATIVE PAYMENTS A payee receives it."
    assert _occurrence_view(source, PAGE, "legacy") == [40.0]
    assert _occurrence_view(source, OTHER_SECTION_PAGE, "legacy") == [40.0, 315.27]


def test_only_the_named_digits_are_blanked_in_place():
    source = (
        HEADER + "2. Count $2 for a unit of 2. See 315.28 F. 3. Stop. "
        "315.27 SPECIAL PAYMENTS FNS unit 1 pays FNS unit 2 the $40."
    )
    recall = sc.authoritative_numeric_recall_text(source, corpus_citation_path=PAGE)
    assert len(recall) == len(source)
    changed = "".join(
        character
        for character, masked in zip(source, recall, strict=True)
        if character != masked
    )
    assert changed.isdigit()
    assert _occurrence_view(source, PAGE, "legacy") == [1.0, 2.0, 2.0, 2.0, 40.0]


# Worked examples are not a recognized shape: every number stays demanded.
# These are the acceptance cases for a later, separate change.
EXAMPLES = (
    ("Count the payment in full. Example: The child receives $600.", [600.0]),
    (
        "Count it. Example: The child receives $600. The standard deduction "
        "is $300 for each unit.",
        [600.0, 300.0],
    ),
    (
        "Count it. Example: The unit pays rent $600 The shelter cap is $300 "
        "for all units.",
        [600.0, 300.0],
    ),
    (
        "Count it. Example: The unit pays $40. If the unit reports within 10 "
        "days, allow $45.",
        [40.0, 10.0, 45.0],
    ),
    ("Count it. Example: A fee of $600 Exception: a limit of $300", [600.0, 300.0]),
    ("Count it. Example: A fee of $600 monthly ii. Sporadic income of $300", []),
    ("Example: The unit pays $300/month.", [300.0]),
    ("Example: The unit pays $300 monthly.", [300.0]),
    ("Example: The members pay $50 each.", [50.0]),
    ("Example: The unit deducts up to $198.", [198.0]),
    ("Example: The worker earns $7.25 at a job.", [7.25]),
    ("Example: Use the 3-month base period for the unit.", [3.0]),
    ("Example: 3 persons share the cost.", [3.0]),
    ("Example: The 20 percent deduction applies to the unit.", []),
    ("EXAMPLE: The member is age 60 and pays $40.", [60.0, 40.0]),
    ("For Example: 3 persons share the $40 cost.", [3.0, 40.0]),
    ("For example, 3 persons share the $40 cost.", [3.0, 40.0]),
    (
        "Example: FNS unit 1 has a child who receives $600.\n\nThe maximum "
        "allotment for FNS unit 2 is $535.",
        [1.0, 600.0, 2.0, 535.0],
    ),
)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(("source", "demanded"), EXAMPLES)
def test_worked_example_numbers_stay_demanded_in_both_gates(source, demanded, profile):
    _assert_both_gates_keep(source, PAGE, profile, demanded)
    recall = sc.authoritative_numeric_recall_text(source, corpus_citation_path=PAGE)
    assert recall == sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=UNSECTIONED_MANUAL
    )


# The decisions of the `See Example N` tests hold on a sectioned page too:
# their fixtures use an unsectioned path, where the rules above are off.
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(("source", "kept"), SEE_EXAMPLE_KEPT)
def test_see_example_decisions_hold_on_a_sectioned_page(source, kept, profile):
    assert _occurrence_view(source, PAGE, profile) == sorted(kept)
    assert _typed_recall(source, PAGE, profile) == sorted(set(kept))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    ("source", "demanded"),
    (
        ("The unit pays $50 and then $1. (See Example 1 )", [1.0]),
        ("The unit pays $50 for 1 month. (See Example 1 )", [1.0]),
        ("Example 1: the household pays $50 and then $1.", [1.0]),
        ("Example 3\nThe household pays $50.", [3.0]),
        ("Example: 3 persons pay $50.", [3.0]),
        ("The unit pays $50. See Example 1 and pay 2.", [1.0, 2.0]),
        ("The unit pays $50. (See Example 1 ) Then pay 2.", [2.0]),
    ),
)
def test_see_example_recall_decisions_hold_on_a_sectioned_page(
    source, demanded, profile
):
    assert _typed_recall(source, PAGE, profile, scalars=(50,)) == demanded
