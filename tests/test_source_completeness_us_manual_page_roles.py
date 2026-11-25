"""Page furniture on a sectioned US manual page is not a rule value.

State SNAP manual pages were refused on numeric recall, and on the encode
eval's occurrence count, for numbers that only organize the page: the running
header (page, section, change notice), the document's own section labels,
list markers flattened into a line, the labels of the units in a narrated
case, and the numbers of a worked example that the page states nowhere else.

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

PAGE = "us-zz/manual/agency/fns/fns-315-special-budgeting/page-16"
OTHER_SECTION_PAGE = "us-zz/manual/agency/fns/fns-340-deductions/page-16"
UNSECTIONED_MANUAL = "us-zz/manual/agency/handbook/shelter-costs/block-3"
REGULATION = "us-zz/regulation/fns-315-special-budgeting/page-16"
PROFILES = ("legacy", "en-US")
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


def _typed_recall(source: str, citation: str, profile: str) -> list[float]:
    """The values typed numeric recall demands of a module with no scalars."""

    candidate = yaml.safe_dump({"format": "rulespec/v1", "module": {}, "rules": []})
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
        TITLE + "16 SecƟon 315 Change #3 2017 September 20, 2017 "
        "Count the payment of $40 in full.",
        [40.0],
    ),
    (
        TITLE + "16 Section 310 Change #6-20XX July 12, 2021 "
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
    # A heading of the document's own section.
    (
        "Count the payment of $40 in full. 315.27 REPRESENTATIVE PAYMENTS "
        "A payee receives the money.",
        [40.0],
    ),
    (
        "The payment of $40 is counted in full.\n\n315.30 STRIKERS A. Count it.",
        [40.0],
    ),
    # A reference to the document's own section.
    ("Count $40 in full. If there was a loss, see 315.28 F. Enter it.", [40.0]),
    ("Count $40 in full. See 315.28 C. for the list.", [40.0]),
    ("Count $40 in full, as shown in 315.02, A. Enter it.", [40.0]),
    ("Count $40 in full. See 315.6, A, 4.\n\nEnter the claim.", [40.0]),
    ("Count $40 in full (see 315.28).", [40.0]),
    # The glyph spelling of `Section` reads as `Section`.
    ("Refer to SecƟon 210, Household Concept, for the $40 limit.", [40.0]),
    ("A person defined in SecƟon 212.02, Person, has a $40 limit.", [40.0]),
    # List markers that open a sentence, a line or the page body.
    (
        "Give the unit time to verify. 2. If the unit paid $40, allow it. "
        "3. Do not ask for more.",
        [40.0],
    ),
    (HEADER + "4. A date must be set. The claim is $40.", [40.0]),
    ("The cost is $50 monthly ($50 for each month). 6. Budget it.", [50.0]),
    ("The shelter is: (a) temporary. 2. A halfway house costs $40.", [40.0]),
    # Unit labels of a narrated case.
    (
        "FNS unit 1 has the child. FNS unit 2 has the parent. Count $40.",
        [40.0],
    ),
)

# A number next to a similar shape keeps its demand.
STILL_DEMANDED = (
    # A header whose page is not the path's page is not this page's header.
    (
        TITLE + "17 Section 315 Change #01-2022 January 11, 2022 Count $40.",
        [17.0, 2022.0, 40.0],
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
    # A row under a quantity caption stays a value.
    ("Benefit amounts, in dollars:\n315.27 REPRESENTATIVE PAYMENTS", [315.27]),
    # Another document's section number is not this document's label.
    ("Count $40. 340.15 SPECIAL PROCEDURES apply.", [40.0, 340.15]),
    ("Count $40. See 340.15 C. for the list.", [40.0, 340.15]),
    # A reference needs `see` or `in` and a closed label.
    ("The unit pays 315.28 F. Enter it.", [315.28]),
    ("See 315.28 dollars of income.", [315.28]),
    ("Count the $40 in 315.28 cases.", [40.0, 315.28]),
    # The glyph rule reads only what `Section` would.
    ("The SecƟon pays $210, and $40 more.", [210.0, 40.0]),
    # A number that ends a sentence is not a list marker.
    ("The limit is a household of 2. If more, deny.", [2.0]),
    ("Count income for children under 6. Advertising costs count.", [6.0]),
    ("Use Form No. 5. Complete it.", [5.0]),
    ("The limit is $2. If more, deny.", [2.0]),
    ("Multiply by 2.15 and then stop. The result is 3.5. Enter it.", [2.15, 3.5]),
    ("Wait. 30 Days must pass.", [30.0]),
    ("Wait. 2. 3 months must pass.", [2.0, 3.0]),
    # One unit label, or a count of units, is not a label set.
    ("FNS unit 2 persons or more get $40.", [2.0, 40.0]),
    ("An FNS unit 1 or 2 persons in size gets $40.", [1.0, 2.0, 40.0]),
    ("FNS unit 2 has the parent. FNS unit 3 has the child.", [2.0, 3.0]),
    ("FNS unit 1 has a child. A unit of 2 gets $40.", [1.0, 2.0, 40.0]),
)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(("source", "demanded"), ACCEPTED)
def test_page_furniture_is_not_demanded_by_either_gate(source, demanded, profile):
    _assert_both_gates(source, PAGE, profile, demanded)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(("source", "demanded"), STILL_DEMANDED)
def test_similar_shape_keeps_its_demand_in_both_gates(source, demanded, profile):
    _assert_both_gates_keep(source, PAGE, profile, demanded)


@pytest.mark.parametrize(
    "citation", (UNSECTIONED_MANUAL, REGULATION, "us-zz/manual/agency/fns/page-16")
)
@pytest.mark.parametrize(("source", "_demanded"), ACCEPTED)
def test_rules_apply_only_on_a_sectioned_manual_page(source, _demanded, citation):
    assert sc._us_manual_page_role_spans(source, corpus_citation_path=citation) == ()
    assert (
        sc._us_manual_example_passage_spans(source, corpus_citation_path=citation) == ()
    )


def test_own_section_comes_from_the_citation_path():
    source = "Count $40. 315.27 REPRESENTATIVE PAYMENTS A payee receives it."
    assert _occurrence_view(source, PAGE, "legacy") == [40.0]
    assert _occurrence_view(source, OTHER_SECTION_PAGE, "legacy") == [40.0, 315.27]


def test_only_the_named_digits_are_blanked_in_place():
    source = (
        HEADER + "2. Count $2 for a unit of 2. See 315.28 F. "
        "315.27 PAYMENTS FNS unit 1 pays FNS unit 2 the $40."
    )
    recall = sc.authoritative_numeric_recall_text(source, corpus_citation_path=PAGE)
    assert len(recall) == len(source)
    changed = "".join(
        character
        for character, masked in zip(source, recall, strict=True)
        if character != masked
    )
    assert changed.isdigit()
    assert _occurrence_view(source, PAGE, "legacy") == [2.0, 2.0, 40.0]


# Worked examples: (source, values still demanded)
EXAMPLES = (
    # Numbers only the example states.
    (
        "Count the payment in full. Example: The child receives $600. The "
        "parent keeps $300 each month.",
        [],
    ),
    (
        "Divide by the months covered. EXAMPLE: A cost of $50 every other "
        "month is $25 monthly ($25 for each month). 6. Budget the cost.",
        [],
    ),
    # The example ends at a line break.
    (
        "Count it. Example: The child receives $600.\n\nThe standard "
        "deduction is $300.",
        [300.0],
    ),
    # The example ends at the next list marker, outline marker, own-section
    # heading or note.
    (
        "Count it. Example: The child receives $600. 2. The standard "
        "deduction is $300.",
        [300.0],
    ),
    (
        "Count it. Example: The child receives $600. B. The standard "
        "deduction is $300.",
        [300.0],
    ),
    (
        "Count it. Example: The child receives $600. 315.27 PAYMENTS The "
        "standard deduction is $300.",
        [300.0],
    ),
    (
        "Count it. Example: The child receives $600. NOTE: The standard "
        "deduction is $300.",
        [300.0],
    ),
    # An example that restates a rule value: every occurrence stays demanded.
    (
        "The standard deduction is $300. Example: The unit has $900 and "
        "the $300 standard deduction leaves $600.",
        [300.0],
    ),
    (
        "Example: The unit has the $300 standard deduction.\n\nThe $300 "
        "standard deduction applies to all units.",
        [300.0],
    ),
    # A number before the example is outside it.
    ("The $300 standard deduction applies. Example: The unit has $900.", [300.0]),
    # Not an example marker: every number stays.
    ("For example, 3 persons share the $40 cost.", [3.0, 40.0]),
    ("Examples of fees include a $40 class fee.", [40.0]),
    ("Example 2: the household pays $40.", [2.0, 40.0]),
    ("See, for example, the $40 fee.", [40.0]),
    ("Counterexample: the household pays $40.", [40.0]),
    # A sentence that names a unit label first introduced in the example is
    # part of the example.
    (
        "Count it. Example: FNS unit 1 has a child who receives $600. FNS "
        "unit 2 has the parent, who keeps $300.\n\nIncome counted in each "
        "case:\n\nFNS unit 1 will have $600 counted. The limit is $40.",
        [40.0],
    ),
    # Unit labels introduced outside an example extend nothing.
    (
        "FNS unit 1 has a child. FNS unit 2 has the parent. Example: The "
        "child receives $600.\n\nFNS unit 1 has $600 counted.",
        [600.0],
    ),
)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(("source", "demanded"), EXAMPLES)
def test_example_only_numbers_are_not_demanded_by_either_gate(
    source, demanded, profile
):
    _assert_both_gates(source, PAGE, profile, demanded)


@pytest.mark.parametrize("profile", PROFILES)
def test_example_value_restated_outside_keeps_every_occurrence(profile):
    source = (
        "The standard deduction is $300. Example: The unit has $900 and "
        "the $300 standard deduction leaves $600."
    )
    assert _occurrence_view(source, PAGE, profile) == [300.0, 300.0]


@pytest.mark.parametrize("profile", PROFILES)
def test_example_filter_is_off_outside_a_sectioned_manual_page(profile):
    source = "Count it. Example: The child receives $600."
    assert _occurrence_view(source, UNSECTIONED_MANUAL, profile) == [600.0]
    assert _typed_recall(source, UNSECTIONED_MANUAL, profile) == [600.0]


def test_example_passage_bounds():
    source = (
        "Count it. Example: The child receives $600. The parent keeps $300. "
        "2. Budget the rest. EXAMPLE: A fee of $40.\nThe limit is $50."
    )
    passages = sc._us_manual_example_passage_spans(source, corpus_citation_path=PAGE)
    assert [source[start:end] for start, end in passages] == [
        "Example: The child receives $600. The parent keeps $300. ",
        "EXAMPLE: A fee of $40.",
    ]


def test_named_scalar_for_an_example_value_is_still_credited():
    """Naming an example value is allowed; it is only not required."""

    source = "Count it. Example: The child receives $600."
    candidate = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {},
            "rules": [{"name": "example_amount", "versions": [{"formula": 600}]}],
        }
    )
    issues = sc.analyze_complete_source_unit(
        candidate,
        source,
        corpus_citation_path=PAGE,
        test_cases=[],
        extract_numeric_occurrences=_extractor("legacy"),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues
    assert not [issue for issue in issues if issue.startswith(PREFIX)]
