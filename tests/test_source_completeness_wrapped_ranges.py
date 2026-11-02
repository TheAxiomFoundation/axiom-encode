"""Wrapped citation ranges and decimal self-references keep their owning branch.

A PDF line break inside a citation must not change the legal paths a source
outline yields. These cover two shapes the Colorado structure fix left open:

- the far end of a ``to`` or ``through`` range that wraps onto its own line
  with no ``of this section`` tail after it; and
- a decimal subsection that repeats its own locator before ``for`` or
  ``but was`` after a lead other than ``this subsection``.

The negative controls keep every real branch that only looks like one of
these shapes: a list introduced by a preposition after a citation, a range
whose ends disagree, a temporal copy after a completed sentence, and a
different decimal root.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CITATION = "us-co/statute/39/39-22-2003"


def _paths(text: str) -> list[tuple[str, ...]]:
    return [
        branch.path
        for branch in sc.recognize_source_structure(text, corpus_citation_path=CITATION)
    ]


def _branch(text: str, path: tuple[str, ...]) -> sc.SourceStructureBranch:
    (branch,) = (
        branch
        for branch in sc.recognize_source_structure(text, corpus_citation_path=CITATION)
        if branch.path == path
    )
    return branch


@pytest.mark.parametrize(
    ("source", "expected", "owner", "range_end"),
    [
        (
            "(4) The amounts in subsection (1)(a) to\n(1)(c) shall be adjusted.\n"
            "(5) The next subsection.",
            [("4",), ("5",)],
            ("4",),
            "(1)(c)",
        ),
        (
            "(4) The amounts in subsections (1.5) to\n(1.7) are adjusted annually.\n"
            "(5) The next subsection.",
            [("4",), ("5",)],
            ("4",),
            "(1.7)",
        ),
        (
            "(4) The amounts in subsections (1) through\n(3) are adjusted.\n"
            "(5) The next subsection.",
            [("4",), ("5",)],
            ("4",),
            "(3)",
        ),
        (
            "(4) The amounts specified in subsections (1) to\n(3) shall be "
            "adjusted.\n(5) The next subsection.",
            [("4",), ("5",)],
            ("4",),
            "(3)",
        ),
        (
            "(4) (a) Amounts in paragraphs (b) to\n(d) are adjusted annually.\n"
            "(b) The second paragraph.\n(5) The next subsection.",
            [("4",), ("4", "a"), ("4", "b"), ("5",)],
            ("4", "a"),
            "(d)",
        ),
        (
            "(4) (a) The rules:\n(I) The amounts in subparagraphs (II) to\n"
            "(IV) apply to the refund.\n(II) The second subparagraph.\n"
            "(b) The second paragraph.\n(5) The next subsection.",
            [
                ("4",),
                ("4", "a"),
                ("4", "a", "i"),
                ("4", "a", "ii"),
                ("4", "b"),
                ("5",),
            ],
            ("4", "a", "i"),
            "(IV)",
        ),
        (
            "(4) The first subsection.\n(5) (a) The amounts in subsections (4) "
            "to\n(5) apply to the refund.\n(b) The second paragraph.\n"
            "(6) The next subsection.",
            [("4",), ("5",), ("5", "a"), ("5", "b"), ("6",)],
            ("5", "a"),
            "(5)",
        ),
    ],
)
def test_wrapped_range_end_stays_in_the_citing_branch(
    source, expected, owner, range_end
):
    assert _paths(source) == expected
    joined = source.replace(" to\n(", " to (").replace(" through\n(", " through (")
    assert _paths(joined) == expected
    assert f"\n{range_end} " in _branch(source, owner).text


def _ordered_roots(integers: int, decimals: frozenset[int]) -> list[str]:
    roots: list[str] = []
    for root in range(1, integers + 1):
        roots.append(str(root))
        if root in decimals:
            roots.append(f"{root}.5")
    return roots


@settings(max_examples=60, deadline=None)
@given(
    integers=st.integers(min_value=3, max_value=7),
    decimals=st.frozensets(st.integers(min_value=1, max_value=7), max_size=3),
    data=st.data(),
    connector=st.sampled_from(["to", "through"]),
    tail=st.sampled_from(
        [
            "are adjusted annually.",
            "shall be adjusted.",
            "apply to the refund.",
            "of this section are adjusted.",
        ]
    ),
    wrap=st.sampled_from(["\n", "\r\n", "\n  "]),
)
def test_wrapping_a_range_end_never_changes_the_outline(
    integers, decimals, data, connector, tail, wrap
):
    roots = _ordered_roots(integers, decimals)
    host = data.draw(st.sampled_from(roots), label="host")
    low, high = sorted(
        data.draw(st.sets(st.sampled_from(roots), min_size=2, max_size=2)),
        key=roots.index,
    )
    lines = []
    for root in roots:
        if root == host:
            lines.append(
                f"({root}) (a) The amounts in subsections ({low}) {connector} "
                f"({high}) {tail}"
            )
            lines.append("(b) The second paragraph.")
        else:
            lines.append(f"({root}) A plain subsection.")
    joined = "\n".join(lines) + "\n"
    wrapped = joined.replace(f" {connector} ({high})", f" {connector}{wrap}({high})")
    assert _paths(wrapped) == _paths(joined)
    assert (host, "b") in _paths(wrapped)


@pytest.mark.parametrize(
    "source",
    [
        # A preposition after a citation can introduce a real list.
        "(1) The executive director shall transfer the amount described in "
        "subsection (3) to\n(a) the general fund; and\n(b) the state education "
        "fund.\n(2) The next subsection.",
        # A range never runs backward, so (a) is a real paragraph.
        "(1) The department shall pay the amount determined under paragraph "
        "(c) to\n(a) the claimant; or\n(b) the claimant's estate.\n"
        "(2) The next subsection.",
        # Nor does it end where it starts.
        "(1) The department shall pay the amount determined under paragraph "
        "(a) to\n(a) the claimant; or\n(b) the claimant's estate.\n"
        "(2) The next subsection.",
        # Without a citation before it, "to" is only a preposition.
        "(1) The department shall pay the refund to\n(a) the claimant; or\n"
        "(b) the claimant's estate.\n(2) The next subsection.",
    ],
)
def test_list_after_a_citation_and_preposition_keeps_its_real_branches(source):
    assert _paths(source) == [("1",), ("1", "a"), ("1", "b"), ("2",)]


@pytest.mark.parametrize(
    "lead",
    [
        "in subsection",
        "under subsection",
        "pursuant to subsection",
        "as described in subsection",
        "this subsection",
    ],
)
@pytest.mark.parametrize(
    "continuation", ["for each lease of a truck", "but was not remitted"]
)
@pytest.mark.parametrize("label", ["1.5", "2.5", "9.5", "11.6"])
def test_wrapped_decimal_self_reference_after_any_subsection_lead_keeps_its_owner(
    lead, continuation, label
):
    root = int(label.split(".")[0])
    source = (
        "(1) Definitions.\n"
        + "".join(f"({index}) A plain subsection.\n" for index in range(2, root + 1))
        + f"({label}) (a) There is allowed a credit {lead}\n({label}) {continuation}.\n"
        "(b) The credit amount is determined by the purchase year.\n"
        f"({root + 1}) The next subsection."
    )
    expected = [
        *((str(index),) for index in range(1, root + 1)),
        (label,),
        (label, "a"),
        (label, "b"),
        (str(root + 1),),
    ]
    assert _paths(source) == expected
    assert f"({label}) {continuation}" in _branch(source, (label, "a")).text


@settings(max_examples=40, deadline=None)
@given(
    root=st.integers(min_value=1, max_value=40),
    decimal=st.integers(min_value=1, max_value=9),
    lead=st.sampled_from(
        ["in subsection", "under subsection", "pursuant to subsection"]
    ),
    continuation=st.sampled_from(
        ["for each lease", "for the tax year", "but was not remitted"]
    ),
    separator=st.sampled_from(["\n", "\r\n", "\n  "]),
)
def test_decimal_self_reference_wrap_never_repeats_its_root(
    root, decimal, lead, continuation, separator
):
    label = f"{root}.{decimal}"
    joined = (
        f"({root}) (a) First.\n({label}) (a) There is allowed a credit {lead} "
        f"({label}) {continuation}.\n(b) The second paragraph.\n"
        f"({root + 1}) The next subsection.\n"
    )
    wrapped = joined.replace(f"{lead} ({label})", f"{lead}{separator}({label})")
    assert _paths(wrapped) == _paths(joined)
    assert _paths(wrapped).count((label,)) == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # A different decimal root after a reference lead is still a root.
        (
            "(1) For purposes of subsection\n(2.5) For each lease the department "
            "must provide requested information.\n(3) The next subsection.",
            [("1",), ("2.5",), ("3",)],
        ),
        # A temporal copy follows a completed sentence.
        (
            "(1) First.\n(2.5) The credit is allowed under subsection.\n"
            "(2.5) For each lease the department must provide requested "
            "information.\n(3) The next subsection.",
            [("1",), ("2.5",), ("2.5",), ("3",)],
        ),
        # Only the bounded "for" and "but was" continuations are references.
        (
            "(1) First.\n(2.5) The credit allowed under subsection\n"
            "(2.5) The department must provide information.\n"
            "(3) The next subsection.",
            [("1",), ("2.5",), ("2.5",), ("3",)],
        ),
    ],
)
def test_decimal_roots_outside_the_self_reference_grammar_remain_branches(
    source, expected
):
    assert _paths(source) == expected
