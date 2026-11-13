"""Compound determiners qualify references only in first-numeric recovery."""

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness

COMPOUNDS = ("each and every", "any and all")
SPACES = (" ", "\t", "\n", "\u00a0")
CHAINS = ("(1)", "(1) and (2)", "(1), (2), and (3)", "(1) and/or (2)(a)")
INTRO = (
    "\n\n".join(f"({chr(n)}) Duty {chr(n)}." for n in range(ord("a"), ord("g")))
    + "\n\n(g) For purposes of this chapter— "
)


def branches(source):
    return completeness.recognize_source_structure(
        source, corpus_citation_path="us/regulation/12/1006/14"
    )


def assert_reference_span_stays_prose(source, chain):
    start = len(INTRO)
    assert not any(
        len(branch.path) > 1 and start <= branch.start < start + len(chain)
        for branch in branches(source)
    )


@pytest.mark.parametrize("compound", COMPOUNDS)
@pytest.mark.parametrize("qualifier", ("of", "in", "under"))
@pytest.mark.parametrize("space", SPACES)
@pytest.mark.parametrize("uppercase", (False, True))
@pytest.mark.parametrize("chain", CHAINS)
def test_compound_reference_tail_stays_prose(
    compound, qualifier, space, uppercase, chain
):
    words = qualifier + " " + compound + " such sections"
    words = words.upper() if uppercase else words
    source = INTRO + chain + space + space.join(words.split()) + " must apply."
    assert_reference_span_stays_prose(source, chain)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    compound=st.sampled_from(COMPOUNDS),
    qualifier=st.sampled_from(("of", "in", "under")),
    space=st.sampled_from(SPACES),
    chain=st.sampled_from(CHAINS),
    unit=st.sampled_from(("section", "paragraph", "clause", "subclause", "schedule")),
)
def test_generated_compound_legal_tail_never_restores_first_numeric(
    compound, qualifier, space, chain, unit
):
    tail = space.join((qualifier + " " + compound + " " + unit).split())
    source = INTRO + chain + space + tail + " applies."
    assert_reference_span_stays_prose(source, chain)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    compound=st.sampled_from(COMPOUNDS),
    qualifier=st.sampled_from(("of", "in", "under")),
    noun=st.sampled_from(
        ("applicant", "agreement", "clauseful type", "sectional arrangements")
    ),
)
def test_compound_without_legal_unit_preserves_operative_first_numeric(
    compound, qualifier, noun
):
    source = INTRO + f"(1) {qualifier} {compound} {noun} is eligible."
    assert any(
        branch.path == ("g", "1") and branch.start == len(INTRO)
        for branch in branches(source)
    )


@pytest.mark.parametrize(
    "body",
    (
        "any qualifying person.",
        "any qualifying person; or (2) any qualifying entity.",
        "under any such agreement is eligible.",
        "each and every applicant is eligible.",
        "any and all applicants are eligible.",
        "of each and every applicant a record is kept.",
        "under each and every agreement is eligible.",
        "under any and all agreements is eligible.",
        "eligible people acting under each and every clause.",
        "of each and every clauseful type.",
        "of any and all sectional arrangements.",
        "in such a qualifying role is eligible.",
    ),
)
def test_substantive_compound_bodies_preserve_first_numeric(body):
    source = INTRO + "(1) " + body
    child = next(branch for branch in branches(source) if branch.path == ("g", "1"))
    assert child.start == len(INTRO)
    assert child.text == source[child.start : child.end].strip()


@pytest.mark.parametrize("compound", COMPOUNDS)
@pytest.mark.parametrize(
    "helper",
    (
        "_headed_first_alpha_reference_tail_starts",
        "_later_heading_reference_tail_starts",
    ),
)
def test_shared_qualifier_consumers_keep_their_original_scope(compound, helper):
    source = (
        "(A) and (B) of " + compound + " clauses."
        if helper == "_headed_first_alpha_reference_tail_starts"
        else "(C) Special rules of " + compound + " clauses."
    )
    assert getattr(completeness, helper)(source) == frozenset()
    assert completeness._FIRST_NUMERIC_TAIL_QUALIFIER.match(
        " of " + compound + " clauses"
    )


def test_compound_extension_changes_only_the_two_diagnosed_alternatives():
    original = completeness._ROOT_CHAPEAU_TAIL_QUALIFIER
    extended = completeness._FIRST_NUMERIC_TAIL_QUALIFIER
    assert extended.pattern == original.pattern.replace(
        r"(?:(?:this|", r"(?:(?:each\s+and\s+every|any\s+and\s+all|this|", 1
    )
    assert extended.flags == original.flags


def test_private_scanner_default_keeps_original_qualifier_scope():
    source = "(1) of each and every clause"
    assert completeness._root_chapeau_reference_tail_starts(source) == frozenset()
    assert completeness._root_chapeau_reference_tail_starts(
        source, qualifier=completeness._FIRST_NUMERIC_TAIL_QUALIFIER
    ) == {0}


def test_only_first_numeric_recovery_explicitly_selects_compound_qualifier():
    with patch.object(
        completeness,
        "_root_chapeau_reference_tail_starts",
        wraps=completeness._root_chapeau_reference_tail_starts,
    ) as scanner:
        assert_reference_span_stays_prose(
            INTRO + "(1) of each and every clause.", "(1)"
        )
    assert scanner.call_count > 0
    assert all(
        call.kwargs == {"qualifier": completeness._FIRST_NUMERIC_TAIL_QUALIFIER}
        for call in scanner.call_args_list
    )


@pytest.mark.parametrize("count", (4, 32, 256))
@pytest.mark.parametrize("qualifies", (False, True))
@pytest.mark.parametrize("compound", COMPOUNDS)
def test_new_qualifier_scans_only_disjoint_label_windows(count, qualifies, compound):
    source = "(1)" + " and (1)" * (count - 1)
    source += (
        " of " + (compound + " ") * count + ("section" if qualifies else "applicant")
    )
    with (
        patch.object(
            completeness,
            "_ROOT_CHAPEAU_TAIL_LABEL",
            wraps=completeness._ROOT_CHAPEAU_TAIL_LABEL,
        ) as labels,
        patch.object(
            completeness,
            "_ROOT_CHAPEAU_TAIL_JOINER",
            wraps=completeness._ROOT_CHAPEAU_TAIL_JOINER,
        ) as joiner,
        patch.object(
            completeness,
            "_FIRST_NUMERIC_TAIL_QUALIFIER",
            wraps=completeness._FIRST_NUMERIC_TAIL_QUALIFIER,
        ) as qualifier,
    ):
        blocked = completeness._root_chapeau_reference_tail_starts(
            source, qualifier=qualifier
        )
    assert blocked == ({index * 8 for index in range(count)} if qualifies else set())
    assert labels.finditer.call_count == 1
    assert joiner.fullmatch.call_count == count - 1
    assert qualifier.match.call_count == count
    for calls in (joiner.fullmatch.call_args_list, qualifier.match.call_args_list):
        assert all(0 <= call.args[1] <= call.args[2] <= len(source) for call in calls)
        assert sum(call.args[2] - call.args[1] for call in calls) <= len(source)
