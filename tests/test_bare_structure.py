"""Bare outlines must retain operative numbers and reject ambiguous labels."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import corpus_resolver
from axiom_encode.bare_structure import (
    bare_outline,
    mask_bare_labels,
    slice_bare_path,
)
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.source_completeness import (
    authoritative_numeric_recall_text,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nz_bare_labels"


def demanded_numbers(source):
    return [
        item.value
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            authoritative_numeric_recall_text(
                source,
                corpus_citation_path="nz/statute/act/public/2018/0032/section/72",
            ),
            profile="en-US",
        )
    ]


@pytest.mark.parametrize(
    ("section", "fragments", "expected_start", "excluded"),
    (
        ("72", ("2", "c"), "the person is aged 65 years", "aged 50 to 64"),
        ("74", ("1", "a"), "if the person’s qualifying benefit", "11 fortnightly"),
    ),
)
def test_verbatim_nz_corpus_child_slicing(section, fragments, expected_start, excluded):
    text = (FIXTURES / f"s{section}.txt").read_text()
    parent = f"nz/statute/act/public/2018/0032/section/{section}"
    sliced = corpus_resolver._slice_parent_body(
        text,
        requested_path=parent + "/" + "/".join(fragments),
        resolved_path=parent,
    )
    assert sliced.startswith(expected_start)
    assert excluded not in sliced


def test_verbatim_s71_unpunctuated_paragraph_stops_at_next_definition():
    text = (FIXTURES / "s71.txt").read_text()
    outline = bare_outline(text)
    assert outline is not None
    assert [label.path for label in outline] == [("a",), ("b",), ("c",)]
    assert slice_bare_path(text, ("c",)) == "veteran’s pension"


@settings(max_examples=100, deadline=None)
@given(
    amount=st.integers(min_value=1, max_value=999_999),
    count=st.integers(min_value=1, max_value=99),
    year=st.integers(min_value=1900, max_value=2099),
    percentage=st.integers(min_value=1, max_value=99),
)
def test_generated_positive_outline_masks_only_labels(amount, count, year, percentage):
    source = (
        "1\nA person is entitled if—\n"
        f"a\nthe amount is ${amount} for {count} children; and\n"
        f"b\nthe rate is {percentage}% from {year}.\n"
        "2\nThe entitlement ends on death.\n"
    )
    outline = bare_outline(source)
    assert outline is not None
    assert tuple(label.path for label in outline) == (
        ("1",),
        ("1", "a"),
        ("1", "b"),
        ("2",),
    )
    masked = mask_bare_labels(source)
    label_offsets = {
        offset for label in outline for offset in range(label.start, label.marker_end)
    }
    for offset, char in enumerate(source):
        assert masked[offset] == (" " if offset in label_offsets else char)
    assert f"${amount}" in masked
    assert f"{count} children" in masked
    assert f"{percentage}%" in masked
    assert str(year) in masked
    assert len(source) == len(masked)
    assert source.count("\n") == masked.count("\n")
    # Calendar years are intentionally excluded by the existing inventory;
    # the mask itself must still preserve every character of the year.
    assert demanded_numbers(source) == [amount, count, percentage]


@settings(max_examples=120, deadline=None)
@given(
    amount=st.integers(min_value=1, max_value=999_999),
    count=st.integers(min_value=1, max_value=99),
    year=st.integers(min_value=1900, max_value=2099),
    percentage=st.integers(min_value=1, max_value=99),
    standalone=st.integers(min_value=1, max_value=999),
    layout=st.sampled_from(("table", "caption", "running", "list")),
    placement=st.sampled_from(("before_sibling", "after_sibling")),
)
def test_generated_mixed_outline_preserves_every_operative_number(
    amount, count, year, percentage, standalone, layout, placement
):
    # Every layout includes a positively operative standalone integer. It may
    # equal a real sibling label: the parser must retain the entire ambiguous
    # outline instead of assuming that an expected ordinal is structural.
    headings = {
        "table": "Table of annual rates\n",
        "caption": "Number of dependent children\n",
        "running": "The prescribed amount is\n",
        "list": "The following counts are payable:\n",
    }
    operative = (
        headings[layout] + f"{standalone}\nThe annual family entitlement is payable.\n"
    )
    sibling = "2\nThe entitlement ends on death.\n"
    tail = operative + sibling if placement == "before_sibling" else sibling + operative
    source = (
        "1\nA person is entitled if—\n"
        f"a\nthe amount is ${amount} for {count} children; and\n"
        f"b\nthe rate is {percentage}% from {year}.\n" + tail
    )
    masked = mask_bare_labels(source)
    assert bare_outline(source) is None
    assert masked == source
    assert f"\n{standalone}\nThe annual family entitlement" in masked
    assert f"${amount}" in masked
    assert f"{count} children" in masked
    assert f"{percentage}%" in masked
    assert str(year) in masked
    assert demanded_numbers(source) == [1, amount, count, percentage] + (
        [standalone, 2] if placement == "before_sibling" else [2, standalone]
    )


@pytest.mark.parametrize("token", ("a", "c", "2"))
@pytest.mark.parametrize("punctuated", (False, True))
def test_repeated_or_out_of_sequence_label_fails_closed(token, punctuated):
    source = (
        "1\nA person is entitled if—\n"
        "a\nthe person meets the first requirement;\n"
        f"{token}\nthe person meets another requirement{';' if punctuated else ''}\n"
        "b\nthe person meets the final requirement.\n"
        "2\nThe entitlement ends on death.\n"
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert slice_bare_path(source, ("1", "a")) is None


@pytest.mark.parametrize(
    "source",
    (
        "1\nThe entitlement is payable.\n2\nThe annual rate follows.\n"
        "Table of dependent children\n3\nNumber of dependent children.\n",
        "1\nThe amount is $100.\n2\nThe amount is $200.\n",
        "1\nThe number of eligible children is 10.\n"
        "2\nThe number of eligible children is 20.\n",
        "1\nFirst dependent child.\n2\nSecond dependent child.\n",
    ),
)
def test_ordered_numeric_table_and_list_rows_stay_operative(source):
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source


@settings(max_examples=60, deadline=None)
@given(
    captions=st.lists(
        st.sampled_from(
            (
                "Eligible children.",
                "Qualified children.",
                "Eligible dependent children.",
                "Qualified dependent children.",
                "Children receiving the benefit.",
            )
        ),
        min_size=1,
        max_size=6,
    )
)
def test_generated_ordered_count_rows_after_an_outline_stay_demanded(captions):
    source = (
        "1\nA person is entitled if—\n"
        "a\nthe person receives a qualifying benefit; and\n"
        "b\nthe person resides here.\n"
        + "".join(f"{index}\n{caption}\n" for index, caption in enumerate(captions, 2))
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert demanded_numbers(source) == [1, *range(2, len(captions) + 2)]


def test_nested_roman_numeric_suffix_and_inserted_letter_paths():
    source = (
        "1\nA person is entitled if—\n"
        "a\nthe person meets either requirement—\n"
        "i\nthe first condition is fulfilled; or\n"
        "ii\nthe second condition is fulfilled; and\n"
        "b\nthe person meets the second requirement; and\n"
        "c\nthe person meets the third requirement; and\n"
        "ca\nan additional third requirement is fulfilled; and\n"
        "d\nthe person meets the final requirement.\n"
        "1A\nAn additional restriction is applicable.\n"
        "2\nThe entitlement ends on death.\n"
    )
    assert slice_bare_path(source, ("1", "a", "ii")) == (
        "the second condition is fulfilled; and"
    )
    assert slice_bare_path(source, ("1", "c")) == (
        "the person meets the third requirement; and"
    )
    assert slice_bare_path(source, ("1A",)) == (
        "An additional restriction is applicable."
    )


def test_multiletter_paragraphs_after_z():
    labels = tuple(chr(ord("a") + index) for index in range(26)) + ("aa", "ab")
    source = "A person is entitled if—\n" + "".join(
        f"{label}\nthe specified condition is fulfilled; or\n" for label in labels
    )
    assert (
        slice_bare_path(source, ("aa",)) == "the specified condition is fulfilled; or"
    )


def test_multiletter_numbering_cannot_return_to_an_earlier_letter():
    labels = tuple(chr(ord("a") + index) for index in range(26)) + ("aa", "b")
    source = "A person is entitled if—\n" + "".join(
        f"{label}\nthe specified condition is fulfilled; or\n" for label in labels
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source


def test_missing_numeric_or_inserted_letter_fails_closed():
    for source in (
        "1\nA person is entitled if—\na\nthe condition is fulfilled.\n"
        "3\nThe entitlement ends on death.",
        "A person is entitled if—\na\nthe first condition is fulfilled; and\n"
        "ac\nthe inserted condition is fulfilled; and\nb\nthe final condition is fulfilled.",
    ):
        assert bare_outline(source) is None
        assert mask_bare_labels(source) == source


def test_parenthetical_body_has_no_bare_outline_or_mask_changes():
    source = (
        "(1) A person is entitled if—\n"
        "(a) the amount is $25 for 3 children; and\n"
        "(b) the rate is 20% from 2026.\n"
        "(2) The entitlement ends on death."
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert re.findall(r"\d+", mask_bare_labels(source)) == re.findall(r"\d+", source)


def test_mixed_parenthetical_and_bare_layout_keeps_existing_slicer():
    source = (
        "1\nA person is entitled if—\na\nthe condition is fulfilled.\n"
        "(2) The entitlement ends on death.\n"
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source


def test_marker_text_containing_means_is_never_an_unlabelled_boundary():
    source = (
        "1\nA person is entitled if—\n"
        "a\nqualifying benefit means a payment under this Act; and\n"
        "b\nthe person receives the specified payment.\n"
        "2\nThe entitlement ends on death.\n"
    )
    assert slice_bare_path(source, ("1", "a")) == (
        "qualifying benefit means a payment under this Act; and"
    )
    paragraph_source = (
        "The following conditions apply—\n"
        "a\nqualifying benefit means a payment under this Act; and\n"
        "b\nthe person receives the specified payment.\n"
    )
    assert slice_bare_path(paragraph_source, ("a",)) == (
        "qualifying benefit means a payment under this Act; and"
    )


@given(
    introduction=st.sampled_from(
        (
            "the number of dependent children is—",
            "the prescribed amount is:",
            "the following counts are payable—",
            "the number of dependent children is as follows.",
            "the prescribed amount is given below;",
            "the annual family entitlement amount.",
        )
    )
)
@settings(max_examples=10, deadline=None)
def test_generated_expected_sibling_count_after_dangling_intro_stays_demanded(
    introduction,
):
    source = (
        "1\nA person is entitled if—\n"
        "a\nthe person receives a qualifying benefit; and\n"
        f"b\n{introduction}\n"
        "2\nThe annual family entitlement is payable.\n"
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert demanded_numbers(source) == [1, 2]


def test_relative_roman_parent_body_slices_by_evidence():
    source = "i\nthe first condition is fulfilled; or\nii\nthe second condition is fulfilled.\n"
    assert slice_bare_path(source, ("ii",)) == "the second condition is fulfilled."


def test_unbounded_digit_line_fails_closed_without_integer_conversion():
    source = "9" * 5000 + "\nA person is entitled if—\na\nthe condition is fulfilled.\n"
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("profile", ("legacy", "en-US", "da-DK"))
@pytest.mark.parametrize("adjacent", (False, True))
def test_numeric_profiles_retain_uncertified_standalone_table_cells(profile, adjacent):
    source = (
        "Table of children\n1\n700\nRate applies.\n"
        if adjacent
        else "Table of children\n1\nEligible children.\n2\nQualified children.\n"
    )
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source, profile=profile
    )
    assert [item.value for item in inventory] == [1, 700 if adjacent else 2]
    assert [source[item.start : item.end] for item in inventory] == [
        "1",
        "700" if adjacent else "2",
    ]


def test_definition_introduced_by_last_child_cannot_move_to_ancestor():
    source = (
        "1\nThe following definitions apply:—\n"
        "qualifying benefit means—\n"
        "a\na main benefit under this Act; or\n"
        "b\nthe pension has the following meaning—\n"
        "minimum period means a period of at least 22 weeks.\n"
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source


def test_four_letter_insertions_participate_in_sibling_boundaries():
    source = "A person is entitled if—\n" + "".join(
        f"{label}\nthe specified condition is fulfilled; or\n"
        for label in ("a", "aa", "aaa", "aaaa", "b")
    )
    assert (
        slice_bare_path(source, ("aaa",)) == "the specified condition is fulfilled; or"
    )
    assert (
        slice_bare_path(source, ("aaaa",)) == "the specified condition is fulfilled; or"
    )


def test_over_cap_or_unknown_lowercase_lines_are_never_silently_skipped():
    for token in ("a" * 33, "unrecognised"):
        source = (
            "1\nA person is entitled if—\n"
            "a\nthe first condition is fulfilled; and\n"
            "b\nthe final condition is fulfilled.\n"
            f"{token}\nThe additional condition applies.\n"
        )
        assert bare_outline(source) is None
        assert mask_bare_labels(source) == source


def test_long_parenthetical_label_blocks_mixed_bare_layout():
    source = (
        "1\nA person is entitled if—\n"
        "a\nthe first condition is fulfilled; and\n"
        "b\nthe final condition is fulfilled.\n"
        "(aaaa) An additional condition applies.\n"
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("ending", ("\n", "\r\n", "\r", "\v", "\u2028"))
def test_definition_boundaries_and_masks_preserve_original_line_endings(ending):
    source = ending.join(
        (
            "qualifying benefit means—",
            "a",
            "a main benefit; or",
            "b",
            "a pension.",
            "winter period means the 22-week period.",
        )
    )
    outline = bare_outline(source)
    assert outline is not None
    assert slice_bare_path(source, ("b",)) == "a pension."
    masked = mask_bare_labels(source, outline)
    assert len(masked) == len(source)
    assert masked.count(ending) == source.count(ending)
    assert "22-week" in masked


def test_sliced_leaf_and_subsection_have_no_own_label_or_glued_label():
    source = (FIXTURES / "s72.txt").read_text()
    leaf = slice_bare_path(source, ("2", "c"))
    assert leaf is not None and leaf.startswith("the person is aged 65 years")
    assert demanded_numbers(leaf) == [65]
    subsection = slice_bare_path(source, ("2",))
    assert subsection is not None and not subsection.startswith("2\n")
    assert bare_outline(subsection) is not None
    assert 2 not in demanded_numbers(subsection)
    branches = recognize_source_structure(subsection)
    assert any(branch.path == ("c",) for branch in branches)
    assert not any("c the person" in branch.text for branch in branches)


def test_singleton_descendant_is_rendered_only_from_parent_certificate():
    source = "1\nA person is entitled if—\na\nthe person is aged 65 years.\n"
    assert bare_outline(source) is not None
    assert (
        bare_outline("A person is entitled if—\na\nthe person is aged 65 years.")
        is None
    )
    sliced = slice_bare_path(source, ("1",))
    assert sliced == "A person is entitled if—\n(a)\nthe person is aged 65 years."
    assert [branch.path for branch in recognize_source_structure(sliced)] == [("a",)]
    assert demanded_numbers(sliced) == [65]
    branches = recognize_source_structure(sliced)
    conditions = sc._source_exception_branches(
        sliced, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert not any("a the person" in item.text for item in conditions)


def test_detached_definition_singleton_projection_fails_closed():
    source = (
        "1\nThe following definitions apply:—\n"
        "qualifying benefit means—\na\na main benefit.\n"
        "minimum period means a period of at least 22 weeks.\n"
    )
    assert bare_outline(source) is not None
    assert slice_bare_path(source, ("1",)) is None
    assert slice_bare_path(source, ("1", "a")) == "a main benefit."


def test_definition_boundary_cannot_leave_later_children_outside_their_ancestor():
    source = (
        "1\nThe following definitions apply:—\n"
        "qualifying benefit means—\n"
        "a\na benefit that qualifies if—\n"
        "i\nthe person is eligible.\n"
        "minimum period means a period of at least 22 weeks.\n"
        "ii\nthe person is resident.\n"
        "b\nanother benefit.\n"
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert slice_bare_path(source, ("1", "a")) is None
    assert 22 in demanded_numbers(source)


@pytest.mark.parametrize(
    "term", ("ACC", "Māori benefit", "COVID-19", "person’s benefit")
)
def test_detached_definition_terms_preserve_case_unicode_digits_and_apostrophes(term):
    source = (
        "benefit means—\na\na main benefit; or\nb\na pension.\n"
        f"{term} means a payment of 22 dollars.\n"
    )
    assert bare_outline(source) is not None
    assert slice_bare_path(source, ("b",)) == "a pension."
    assert demanded_numbers(source)[-1] == 22
    definition = next(
        branch for branch in recognize_source_structure(source) if branch.label == term
    )
    assert definition.path == ()
    assert definition.text == source[definition.start : definition.end]


def test_verbatim_capitalized_definition_stops_the_preceding_paragraph():
    source = (FIXTURES / "schedule1_part8_clause93.txt").read_text()
    assert bare_outline(source) is not None
    assert slice_bare_path(source, ("b",)) == "by Part 1 of the Amendment Act"
    assert {"amendments", "Amendment Act", "commencement"} <= {
        branch.label
        for branch in recognize_source_structure(source)
        if branch.kind == "definition"
    }


@pytest.mark.parametrize("term", ('"special benefit"', "special_benefit", "§ benefit"))
@pytest.mark.parametrize("ending", ("\n", "\r", "\u2028"))
def test_unknown_detached_definition_header_abstains_without_losing_numbers(
    term, ending
):
    source = ending.join(
        (
            "benefit means—",
            "a",
            "a main benefit; or",
            "b",
            "a pension.",
            f"{term} means a payment of 22 dollars.",
        )
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert slice_bare_path(source, ("b",)) is None
    assert 22 in demanded_numbers(source)


@settings(max_examples=12, deadline=None)
@given(amount=st.integers(min_value=1, max_value=999))
def test_generated_formula_variable_declarations_cannot_be_sliced_or_masked(amount):
    source = (
        "1\nThe amount is calculated as a + b.\nwhere—\n"
        f"a\nis the annual income of ${amount}; and\n"
        "b\nis the total annual allowance of $25.\n"
    )
    assert bare_outline(source) is None
    assert slice_bare_path(source, ("1", "a")) is None
    assert mask_bare_labels(source) == source
    assert demanded_numbers(source) == [1, amount, 25]
