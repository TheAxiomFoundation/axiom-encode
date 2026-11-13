"""Closed numeric headings have no operative text of their own."""

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness

HEADINGS = (
    "General rule",
    "General rules",
    "Special rule",
    "Special rules",
    "Definition",
    "Definitions",
    "In general",
)
OPERATIVE = (
    "Only residents qualify.",
    "Income shall be reduced by 50.",
    "A claimant must elect relief.",
    "Except that nonresidents do not qualify.",
)
# Same verbatim tax paragraph and synthetic preceding root as the conjunction tests.
TAX_SOURCE = (
    "(1) Preceding root.\n\n"
    "(2) Special rules (A) Regular tax In applying section 1(h) for purposes of "
    "determining the tax under paragraph (1)(A)(i) for any taxable year in which, "
    "without regard to this subsection, the taxpayer’s net capital gain exceeds "
    "taxable income (hereafter in this subparagraph referred to as the capital gain "
    "excess)— (i) the taxpayer’s net capital gain (determined without regard to "
    "section 1(h)(11)) shall be reduced (but not below zero) by such capital gain "
    "excess, (ii) the taxpayer’s qualified dividend income shall be reduced by so much "
    "of such capital gain excess as exceeds the taxpayer’s net capital gain "
    "(determined without regard to section 1(h)(11) and the reduction under clause "
    "(i)), and (iii) adjusted net capital gain, unrecaptured section 1250 gain, and "
    "28-percent rate gain shall each be determined after increasing the amount "
    "described in section 1(h)(4)(B) by such capital gain excess. (B) Alternative "
    "minimum tax In applying section 55(b)(3) for purposes of determining the tax "
    "under paragraph (1)(B)(i) for any taxable year in which, without regard to this "
    "subsection, the taxpayer’s net capital gain exceeds the taxable excess (as "
    "defined in section 55(b)(1)(B))— (i) the rules of subparagraph (A) shall apply, "
    "except that such subparagraph shall be applied by substituting “the taxable "
    "excess (as defined in section 55(b)(1)(B))” for “taxable income”, and (ii) the "
    "reference in section 55(b)(3)(B) to the excess described in section 1(h)(1)(B), "
    "and the reference in section 55(b)(3)(C)(ii) to the excess described in section "
    "1(h)(1)(C)(ii), shall each be treated as a reference to each such excess as "
    "determined under the rules of subparagraph (A) for purposes of determining the "
    "tax under paragraph (1)(A)(i). (C) Definitions Terms used in this paragraph which "
    "are also used in section 1(h) shall have the respective meanings given such terms "
    "by section 1(h), except that in applying subparagraph (B) the adjustments under "
    "part VI of subchapter A shall be taken into account."
)


def tax_parent(source):
    branches = completeness.recognize_source_structure(
        source, corpus_citation_path="us/statute/26/911/f"
    )
    return next(branch for branch in branches if branch.path == ("2",)), branches


def container(prefix, *, label="(2)", kind="paragraph", child_label="(A)"):
    source = label + prefix + child_label + " The taxpayer must elect relief."
    parent = completeness.SourceStructureBranch(
        (label.strip("()."),), kind, label, source, 0, len(source)
    )
    start = len(label + prefix)
    child = completeness.SourceStructureBranch(
        (*parent.path, "a"), "letter", child_label, source[start:], start, len(source)
    )
    return source, parent, child


def is_container(parent, children, source):
    return completeness._is_marker_only_container(
        parent, branches=(parent, *children), source_text=source
    )


@pytest.mark.parametrize("heading", HEADINGS)
def test_closed_heading_before_real_tax_A_has_no_own_operative_text(heading):
    source = TAX_SOURCE.replace("Special rules", heading, 1)
    parent, branches = tax_parent(source)
    first = min(
        (branch for branch in branches if branch.path[:-1] == parent.path),
        key=lambda branch: branch.start,
    )
    assert first.path == ("2", "a") and first.label == "(A)"
    assert source[parent.start : first.start] == "(2) " + heading + " "
    assert completeness._is_marker_only_container(
        parent, branches=branches, source_text=source
    )


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    heading=st.sampled_from(HEADINGS),
    number=st.integers(min_value=1, max_value=999),
    space=st.sampled_from((" ", "\t", " \t ")),
    uppercase=st.booleans(),
)
def test_closed_heading_property_preserves_numeric_and_horizontal_space_scope(
    heading, number, space, uppercase
):
    heading = heading.upper() if uppercase else heading
    source, parent, child = container(space + heading + space, label=f"({number})")
    assert is_container(parent, (child,), source)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(heading=st.sampled_from(HEADINGS), operative=st.sampled_from(OPERATIVE))
def test_operative_insertion_always_requires_parent_coverage(heading, operative):
    source, parent, child = container(" " + heading + " " + operative + " ")
    assert not is_container(parent, (child,), source)


@pytest.mark.parametrize(
    "prefix",
    (
        " 'Special rules' ",
        ' "Special rules" ',
        " Special rules: ",
        " Special rules. ",
        " Special rules\n",
        "\nSpecial rules ",
        " Other rules ",
        " Special rules (A) described in paragraph ",
    ),
)
def test_nonclosed_or_reference_prefixes_remain_accountable(prefix):
    source, parent, child = container(prefix)
    assert not is_container(parent, (child,), source)


@pytest.mark.parametrize(
    "label,kind",
    [("(b)", "letter"), ("(b)", "paragraph"), ("2.", "paragraph"), ("(2)", "letter")],
)
def test_heading_addition_requires_numeric_paragraph_parent(label, kind):
    source, parent, child = container(" Special rules ", label=label, kind=kind)
    assert not is_container(parent, (child,), source)


def test_heading_requires_first_actual_direct_A_at_exact_heading_boundary():
    source, parent, child = container(" Special rules ")
    assert not is_container(parent, (), source)
    assert not is_container(parent, (replace(child, label="(B)"),), source)
    assert not is_container(parent, (replace(child, start=child.start + 1),), source)
    assert not is_container(parent, (replace(child, path=("2", "a", "i")),), source)
    assert not is_container(parent, (replace(child, start=parent.end),), source)
    assert not is_container(parent, (replace(child, path=("3", "a")),), source)
    earlier = replace(child, label="(B)", start=child.start - 1)
    assert not is_container(parent, (child, earlier), source)
    assert is_container(parent, (child,), source)


def test_real_C_and_Bii_remain_operative_after_heading_exemption():
    _parent, branches = tax_parent(TAX_SOURCE)
    for path in (("2", "c"), ("2", "b", "ii")):
        branch = next(branch for branch in branches if branch.path == path)
        assert not completeness._is_marker_only_container(
            branch, branches=branches, source_text=TAX_SOURCE
        )


@pytest.mark.parametrize(
    "source",
    [
        "(2) Special rules",
        "(2) Special rules\n(B) The taxpayer must elect relief.",
        "(2) Special rules (A) and (B) of this section apply.",
    ],
)
def test_real_leaf_first_B_and_reference_tail_remain_accountable(source):
    parent, branches = tax_parent(source)
    assert not completeness._is_marker_only_container(
        parent, branches=branches, source_text=source
    )
