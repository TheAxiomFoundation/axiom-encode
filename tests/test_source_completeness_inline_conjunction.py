"""Conjunctions separate operative items but also occur in legal references."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness.source_completeness import recognize_source_structure

JOINERS = ("; or ", "; and ", ", or ", ", and ")
ROMANS = ("i", "ii", "iii", "iv", "v")
CITATION = "us/regulation/12/1006/14"


def outline(body, *, parent="g", numeric_parent=False):
    prefix = "\n\n".join(
        f"({chr(index)}) Duty {chr(index)}." for index in range(ord("a"), ord(parent))
    )
    child = "\n(1)" if numeric_parent else " Conditions: "
    return prefix + f"\n\n({parent}) Duty {parent}." + child + body


def structural(source):
    return [
        branch
        for branch in recognize_source_structure(source, corpus_citation_path=CITATION)
        if branch.kind in {"paragraph", "letter"}
    ]


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize("parent,numeric_parent", [("g", False), ("h", True)])
def test_conjoined_inline_item_retains_path_and_source_offset(
    joiner, parent, numeric_parent
):
    # A numeric child makes the Roman level unambiguous after (h); the direct
    # (h)/(i) ambiguity is the separate hierarchy change in PR #2115.
    source = outline(
        "(i) one; (ii) two" + joiner + "(iii) three.",
        parent=parent,
        numeric_parent=numeric_parent,
    )
    prefix = (parent, "1") if numeric_parent else (parent,)
    items = [branch for branch in structural(source) if branch.path == (*prefix, "iii")]
    assert len(items) == 1
    assert items[0].start == source.index("(iii)")
    assert items[0].text.startswith("(iii) three.")


@settings(max_examples=80, deadline=None)
@given(
    count=st.integers(min_value=2, max_value=5),
    joiners=st.lists(st.sampled_from(JOINERS), min_size=4, max_size=4),
)
def test_generated_inline_lists_preserve_every_item(count, joiners):
    parts = [
        f"({label}) Condition number {index}."
        for index, label in enumerate(ROMANS[:count])
    ]
    body = parts[0] + "".join(joiner + part for joiner, part in zip(joiners, parts[1:]))
    source = outline(body)
    branches = structural(source)
    expected = [("g", label) for label in ROMANS[:count]]
    assert [branch.path for branch in branches if len(branch.path) > 1] == expected
    for branch, label in zip(branches[-count:], ROMANS[:count], strict=True):
        assert branch.start == source.index(f"({label})")


@settings(max_examples=80, deadline=None)
@given(
    joiner=st.sampled_from((*JOINERS, " or ", " and ")),
    phrase=st.sampled_from(
        (
            "identified in paragraph (b)(2)(i){joiner}(ii) of this section",
            "eligible under (A){joiner}(B)",
            "described in clause (i){joiner}(ii)",
            "described in paragraphs: (i){joiner}(ii) of this section",
            "defined under clause (i){joiner}(ii) of this section",
        )
    ),
)
def test_generated_reference_chains_never_create_branches(joiner, phrase):
    reference = phrase.format(joiner=joiner)
    source = outline(
        "(i) A person "
        + reference
        + " must comply; (ii) second operative condition; (iii) third operative condition."
    )
    branches = structural(source)
    start = source.index(reference)
    assert not any(
        start <= branch.start < start + len(reference) for branch in branches
    )
    operative_starts = {
        source.index("(i) A person"),
        source.index("(ii) second"),
        source.index("(iii) third"),
    }
    assert all(
        branch.start in operative_starts for branch in branches if len(branch.path) > 1
    )


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize("reference_tail", ["", " of this section"])
def test_reference_at_end_of_item_does_not_hide_next_operative_item(
    joiner, reference_tail
):
    source = outline(
        "(i) A person follows the rule in clause (i)"
        + reference_tail
        + joiner
        + "(ii) second operative condition; (iii) third operative condition."
    )
    branches = structural(source)
    assert [branch.path for branch in branches if len(branch.path) > 1] == [
        ("g", "i"),
        ("g", "ii"),
        ("g", "iii"),
    ]
    assert next(
        branch.start for branch in branches if branch.path == ("g", "ii")
    ) == source.index("(ii) second")


@pytest.mark.parametrize(
    "body",
    [
        "must satisfy a separate condition",
        "is exempt from that requirement",
        "has a separate permission",
    ],
)
def test_conjoined_operative_predicate_after_reference_stays_an_item(body):
    source = outline(
        "(i) meets the requirement in clause (i); or (ii) "
        + body
        + "; (iii) meets a third condition."
    )
    assert any(
        branch.path == ("g", "ii") and branch.start == source.index("(ii)")
        for branch in structural(source)
    )


@pytest.mark.parametrize("joiner", JOINERS)
@pytest.mark.parametrize(
    "phrase",
    [
        "described in clause (i){joiner}(ii) must comply.",
        "identified in paragraph (b)(2)(i){joiner}(ii) of this section.",
        "eligible under (A){joiner}(B).",
    ],
)
def test_terminal_reference_without_a_later_duplicate_stays_prose(joiner, phrase):
    reference = phrase.format(joiner=joiner)
    source = outline("(i) A person " + reference)
    start = source.index(reference)
    assert not any(start <= branch.start for branch in structural(source))


@settings(max_examples=80, deadline=None)
@given(
    count=st.integers(min_value=2, max_value=5),
    joiner=st.sampled_from(JOINERS),
    head=st.sampled_from(("paragraph (b)(2)", "clause ", "under ")),
)
def test_generated_long_reference_chains_never_add_items(count, joiner, head):
    reference = head + joiner.join(f"({label})" for label in ROMANS[:count])
    source = outline(
        "(i) A person described in " + reference + " of this section must comply."
    )
    start = source.index(reference)
    assert not any(start <= branch.start for branch in structural(source))


@pytest.mark.parametrize("joiner", JOINERS)
def test_conjoined_capital_letter_keeps_its_proven_predecessor(joiner):
    source = outline(
        "(i) Qualifying categories: (A) First category"
        + joiner
        + "(B) Second category."
    )
    branches = structural(source)
    assert [
        (branch.path, branch.start) for branch in branches if len(branch.path) > 2
    ] == [
        (("g", "i", "a"), source.index("(A)")),
        (("g", "i", "b"), source.index("(B)")),
    ]
