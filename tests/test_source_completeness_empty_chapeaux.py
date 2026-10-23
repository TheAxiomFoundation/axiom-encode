"""Conditional list chapeaux do not invent independently toggleable gates."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CITATION = "nz/statute/act/public/2018/0032/section/72"
# Social Security Act 2018, s 72(1) and the chapeau of s 72(2), with
# s 72(2)(a). The child text supplies the predicates introduced by each 'if'.
SOURCE = """1
A person is entitled to a winter energy payment for 1 or more days during the winter period if, for those days,—
a
the person is receiving a qualifying benefit (or a portion of it), either as the person granted that benefit or as the spouse or partner of that person; and
b
the qualifying benefit (or a portion of it) is payable to the person; and
c
the person is not disqualified under subsection (2).
2
However, the person is not entitled to a winter energy payment for 1 or more days during the winter period if, for those days,—
a
the qualifying benefit is payable at a rate prescribed in respect of a person who is in a relationship, and MSD has determined under subsection (3) that the person’s spouse or partner is entitled to a winter energy payment.
"""


def _exceptions(source, *, recognize=True):
    branches = (
        sc.recognize_source_structure(source, corpus_citation_path=CITATION)
        if recognize
        else ()
    )
    return sc._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )


def test_nz_parent_condition_chapeaux_add_no_independent_paired_obligations():
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    assert {("1", "a"), ("1", "b"), ("1", "c"), ("2", "a")} <= {
        branch.path for branch in branches
    }
    assert not _exceptions(SOURCE)


@pytest.mark.parametrize(
    "source",
    [
        "A person is entitled to a payment if, for those days,—",
        "A person is entitled to a payment if the person is receiving a benefit: ",
        "A person is entitled to a payment if, for those days,— the person is receiving a benefit.",
    ],
)
def test_unresolved_or_substantive_condition_keeps_paired_obligation(source):
    obligations = _exceptions(source, recognize=False)
    assert obligations
    assert all(
        sc._source_exception_requires_paired_witness(x.text) for x in obligations
    )


def test_only_a_sibling_does_not_resolve_an_empty_chapeau():
    source = (
        "1\nA person is entitled to a payment if, for those days,—\n"
        "2\nA payment has the following rates:—\na\nthe single rate; or\n"
        "b\nthe couple rate.\n"
    )
    second = source.index("2\n")
    first_child = source.index("a\n", second)
    last_child = source.index("b\n", first_child)
    branches = (
        sc.SourceStructureBranch(
            ("1",), "subsection", "(1)", source[:second], 0, second
        ),
        sc.SourceStructureBranch(
            ("2",), "subsection", "(2)", source[second:], second, len(source)
        ),
        sc.SourceStructureBranch(
            ("2", "a"),
            "paragraph",
            "(a)",
            source[first_child:last_child],
            first_child,
            last_child,
        ),
        sc.SourceStructureBranch(
            ("2", "b"),
            "paragraph",
            "(b)",
            source[last_child:],
            last_child,
            len(source),
        ),
    )
    obligations = sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert len(obligations) == 1
    assert obligations[0].path == ("1",)


def test_inactive_children_do_not_resolve_an_empty_chapeau():
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    active = tuple(branch for branch in branches if len(branch.path) == 1)
    obligations = sc._source_exception_branches(
        SOURCE,
        branches=branches,
        active_branches=active,
        deferred_paths=set(),
    )
    assert {branch.path for branch in obligations} == {("1",), ("2",)}


def test_substantive_parent_condition_keeps_paired_obligation_with_children():
    source = SOURCE.replace(
        "if, for those days,—", "if the person is ordinarily resident,—", 1
    )
    obligations = _exceptions(source)
    assert len(obligations) == 1
    assert obligations[0].path == ("1",)
    assert "ordinarily resident" in obligations[0].text


def test_explicit_child_condition_still_requires_paired_evidence():
    source = SOURCE.replace(
        "the person is not disqualified under subsection (2).",
        "the person qualifies if ordinarily resident.",
    )
    obligations = _exceptions(source)
    assert len(obligations) == 1
    assert obligations[0].path == ("1", "c")
    assert sc._source_exception_requires_paired_witness(obligations[0].text)


@given(
    cue=st.sampled_from(["if", "when", "IF", "When"]),
    adjunct=st.sampled_from(
        ["", ", for those days,", ", in that period,", ", during those months,"]
    ),
    terminator=st.sampled_from([":", "—", "–", "-"]),
    substantive=st.sampled_from(
        [
            "the person is resident",
            "income is below the limit",
            "the benefit is payable",
        ]
    ),
)
def test_empty_chapeau_recognition_never_discards_a_condition_predicate(
    cue, adjunct, terminator, substantive
):
    assert sc._source_condition_is_empty_list_chapeau(f"{cue}{adjunct}{terminator}")
    assert not sc._source_condition_is_empty_list_chapeau(
        f"{cue}{adjunct} {substantive}{terminator}"
    )
