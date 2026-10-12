"""Colorado source paths survive PDF wrapping and shared repeal locators."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CITATION = "us-co/statute/39/39-22-2003"
WRAPPED_REFERENCE = """\
(4) (a) The refund shall be the same as provided in subsection
(3) of this section; except that the executive director shall adjust:
(I) The amount of adjusted gross income.
(b) Use the most recent estimate.
(5) The next subsection.
"""
DECIMAL_SUBSECTIONS = """\
(1) (a) Qualified individual means:
(I) A natural person.
(1.5) Adjusted gross income means:
(a) Federal adjusted gross income.
(1.7) As used in this section:
(a) Excess state revenues has the meaning in section 39-22-2002 (3).
(2) A refund is allowed.
(9.5) Section 39-21-108 (3) applies to the refund.
(10) The department shall provide requested information.
"""
COMBINED_REPEAL = """\
(1) (b) Qualified individual does not include:
(I) Any natural person convicted of a felony.
(II) and (III) Repealed.
(2) A refund is allowed.
(8) and (9) Repealed.
(10) The department shall provide requested information.
"""


def _branches(text):
    return sc.recognize_source_structure(text, corpus_citation_path=CITATION)


def test_wrapped_reference_keeps_its_owning_subsection():
    branches = _branches(WRAPPED_REFERENCE)
    assert [branch.path for branch in branches] == [
        ("4",),
        ("4", "a"),
        ("4", "a", "i"),
        ("4", "b"),
        ("5",),
    ]
    assert "(3) of this section" in branches[1].text


@pytest.mark.parametrize(
    "reference",
    [
        " (2) or\n(5)(a)",
        "\n(2) or\n(5)(a)",
        " (2) or (5)\n(a)",
        "\n(2)\nor\n(5)\n(a)",
        " (2),\n(5)(a)",
        "\n(2), and\n(5)(a)",
    ],
)
def test_wraps_inside_a_cross_reference_keep_all_locators_in_the_owner(reference):
    source = (
        "(4) (a) The refund is as provided in subsection"
        + reference
        + " of this section; the executive director shall adjust:\n"
        "(I) The amount of adjusted gross income.\n"
        "(b) Use the most recent estimate.\n(5) The next subsection."
    )
    assert [branch.path for branch in _branches(source)] == [
        ("4",),
        ("4", "a"),
        ("4", "a", "i"),
        ("4", "b"),
        ("5",),
    ]


@pytest.mark.parametrize(
    "reference",
    [
        "\n(a) to (f)",
        " (a) to\n(f)",
        "\n(a)\nto\n(f)",
        "\n(a) through\n(f)",
    ],
)
def test_wrapped_citation_range_retains_the_owning_paragraph(reference):
    # C.R.S. 39-22-601(6)(g) cites the preceding paragraphs as a range.
    source = (
        "(6) (g) Any assessment made by reason of the provisions of paragraphs"
        + reference
        + " of this subsection (6) shall be limited to federal adjustments.\n"
        "(h) The department must provide requested information.\n"
        "(7) The next subsection."
    )
    branches = _branches(source)
    assert [branch.path for branch in branches] == [
        ("6",),
        ("6", "g"),
        ("6", "h"),
        ("7",),
    ]
    assert reference.strip() in branches[1].text


@pytest.mark.parametrize("label", ["2.3", "2.5", "4.5", "9.5", "11.6"])
def test_wrapped_decimal_self_reference_for_credit_retains_its_owner(label):
    # C.R.S. 39-22-516.8 repeats each credit's own decimal subsection locator.
    next_root = str(int(label.split(".")[0]) + 1)
    source = (
        "(1) Definitions.\n"
        f"({label}) (a) There is allowed a credit in an amount set forth in "
        "paragraph (b) of this subsection\n"
        f"({label}) for each lease of a truck during the tax year.\n"
        "(b) The credit amount is determined by the purchase year.\n"
        f"({next_root}) The next subsection."
    )
    branches = _branches(source)
    assert [branch.path for branch in branches] == [
        ("1",),
        (label,),
        (label, "a"),
        (label, "b"),
        (next_root,),
    ]
    assert f"({label}) for each lease" in branches[2].text


@pytest.mark.parametrize(
    "reference",
    [
        "section 39-1-104\n(10.2) and that is used to determine actual value",
        "section 39-1-104\n(10.2) and which is used to determine actual value",
        "section 20\n(8)(c) of said article X in the intervening year",
    ],
)
def test_wrapped_named_provision_reference_keeps_the_complete_control_clause(reference):
    clause = (
        "(b) An equalization factor shall apply as prescribed in " + reference + "."
    )
    source = "(1) Definitions.\n(2) (a) Earlier rule.\n" + clause + "\n(3) Next rule."
    branches = _branches(source)
    assert [branch.path for branch in branches] == [
        ("1",),
        ("2",),
        ("2", "a"),
        ("2", "b"),
        ("3",),
    ]
    assert (
        next(branch for branch in branches if branch.path == ("2", "b")).text == clause
    )


def test_wrapped_decimal_self_reference_but_keeps_the_complete_control_clause():
    clause = (
        "(b) The tax liability withheld pursuant to paragraph (a) of this subsection\n"
        "(4.5) but was not remitted by the first purchaser shall be deemed satisfied."
    )
    source = "(1) Definitions.\n(4.5) (a) Earlier rule.\n" + clause + "\n(5) Next rule."
    branches = _branches(source)
    assert [branch.path for branch in branches] == [
        ("1",),
        ("4.5",),
        ("4.5", "a"),
        ("4.5", "b"),
        ("5",),
    ]
    assert (
        next(branch for branch in branches if branch.path == ("4.5", "b")).text
        == clause
    )


@pytest.mark.parametrize(
    "body", ["but was not remitted", "But the department must report"]
)
def test_but_continuation_does_not_hide_a_different_decimal_root(body):
    source = (
        "(1) For purposes of this subsection\n"
        f"(2.5) {body} to the treasurer.\n(3) Next rule."
    )
    assert [branch.path for branch in _branches(source)] == [("1",), ("2.5",), ("3",)]


@pytest.mark.parametrize("boundary", [".\n", "\n\n", "\r\n\r\n"])
@pytest.mark.parametrize(
    "reference", ["section 39-1-104", "section 20", "this subsection"]
)
def test_new_reference_tails_do_not_cross_completed_sentences_or_paragraphs(
    boundary, reference
):
    source = (
        "(1) Review "
        + reference
        + boundary
        + "(2.5) And that is used to determine the credit.\n(3) Next rule."
    )
    assert [branch.path for branch in _branches(source)] == [("1",), ("2.5",), ("3",)]


@settings(max_examples=36, deadline=None)
@given(
    tokens=st.sampled_from(
        [
            ["section", "39-1-104", "(10.2)", "and", "that", "is", "used"],
            ["section", "39-1-104", "(10.2)", "and", "which", "is", "used"],
            ["section", "20", "(8)", "(c)", "of", "said", "article", "X"],
            ["this", "subsection", "(4.5)", "but", "was", "not", "remitted"],
        ]
    ),
    breaks=st.sets(st.integers(min_value=0, max_value=7), max_size=7),
    separator=st.sampled_from([" ", "\n", "\n  ", "\r\n"]),
)
def test_wrapped_relative_citation_continuations_keep_the_entire_source_clause(
    tokens, breaks, separator
):
    reference = "".join(
        (separator if index in breaks else " ") + token
        for index, token in enumerate(tokens)
    )
    clause = "(b) The amount shall be determined pursuant to" + reference + "."
    source = "(1) Definitions.\n(4.5) (a) Earlier rule.\n" + clause + "\n(5) Next rule."
    branches = _branches(source)
    assert [branch.path for branch in branches] == [
        ("1",),
        ("4.5",),
        ("4.5", "a"),
        ("4.5", "b"),
        ("5",),
    ]
    assert (
        next(branch for branch in branches if branch.path == ("4.5", "b")).text
        == clause
    )


@pytest.mark.parametrize("body", ["For each lease", "for each lease", "For purposes"])
def test_for_continuation_does_not_hide_a_new_decimal_subsection(body):
    source = (
        "(1) For purposes of this subsection\n"
        f"(2.5) {body} the department must provide requested information.\n"
        "(3) The next subsection."
    )
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {("1",), ("2.5",), ("3",)}
    assert "must provide requested information" in branches[("2.5",)].text


def test_decimal_subsections_have_separate_roots_and_children():
    assert [branch.path for branch in _branches(DECIMAL_SUBSECTIONS)] == [
        ("1",),
        ("1", "a"),
        ("1", "a", "i"),
        ("1.5",),
        ("1.5", "a"),
        ("1.7",),
        ("1.7", "a"),
        ("2",),
        ("9.5",),
        ("10",),
    ]


@pytest.mark.parametrize("child_separator", [" ", "\n"])
def test_decimal_root_after_numbered_child_keeps_its_lower_alpha_child(child_separator):
    source = (
        "(1) (a) First:\n(I) First:\n(A) First:\n"
        "(1) The claimant must file a return.\n"
        "(1.5)"
        + (" Adjusted gross income means:" if child_separator == "\n" else "")
        + child_separator
        + "(a) A new decimal subsection.\n"
        "(2) A new root subsection."
    )
    assert [branch.path for branch in _branches(source)] == [
        ("1",),
        ("1", "a"),
        ("1", "a", "i"),
        ("1", "a", "i", "a"),
        ("1", "a", "i", "a", "1"),
        ("1.5",),
        ("1.5", "a"),
        ("2",),
    ]


def test_decimal_self_reference_after_numbered_child_does_not_repeat_root():
    source = (
        "(1) Definitions.\n(2.5) (a) First:\n(I) First:\n(A) First:\n"
        "(1) The claimant must file a return.\n"
        "(b) There is a credit under this subsection\n"
        "(2.5) for each lease.\n(3) The next subsection."
    )
    assert [branch.path for branch in _branches(source)] == [
        ("1",),
        ("2.5",),
        ("2.5", "a"),
        ("2.5", "a", "i"),
        ("2.5", "a", "i", "a"),
        ("2.5", "a", "i", "a", "1"),
        ("2.5", "b"),
        ("3",),
    ]


@settings(max_examples=32, deadline=None)
@given(
    root=st.integers(min_value=1, max_value=98),
    decimal=st.integers(min_value=1, max_value=9),
    depth=st.integers(min_value=0, max_value=6),
    child_separator=st.sampled_from([" ", "\n"]),
    separator=st.sampled_from([" ", "\n", "\n  ", "\r\n"]),
)
def test_decimal_root_and_self_reference_keep_nested_numbered_content(
    root, decimal, depth, child_separator, separator
):
    label = f"{root}.{decimal}"
    children = ["i", "A", "1", "i", "A", "1"][:depth]
    nested = "".join(
        f"({child}) The claimant must file a return.\n" for child in children
    )
    source = (
        f"({root}) (a) First:\n"
        + nested
        + f"({label})"
        + (" The decimal credit:" if child_separator == "\n" else "")
        + child_separator
        + "(a) The decimal credit:\n"
        + nested
        + "(b) There is a credit under this subsection"
        + separator
        + f"({label}) for each lease.\n({root + 1}) The next subsection."
    )
    expected = [
        (str(root),),
        (str(root), "a"),
        *(
            (str(root), "a", *(child.lower() for child in children[:index]))
            for index in range(1, depth + 1)
        ),
        (label,),
        (label, "a"),
        *(
            (label, "a", *(child.lower() for child in children[:index]))
            for index in range(1, depth + 1)
        ),
        (label, "b"),
        (str(root + 1),),
    ]
    assert [branch.path for branch in _branches(source)] == expected


@pytest.mark.parametrize("outer", ["", "A. Outer section.\n"])
def test_explicitly_attached_decimal_numbered_child_stays_nested(outer):
    source = (
        outer + "(1) Definitions.\n(2.5) (a) First:\n(I) First:\n(A) First:\n"
        "(1) The claimant must file a return.\n"
        "(2.5)(a)(I)(A)(1.5) The claimant must submit an annual report.\n"
        "(3) The next subsection." + ("\nB. Another outer section." if outer else "")
    )
    branches = _branches(source)
    expected = (*(("a",) if outer else ()), "2.5", "a", "i", "a", "1.5")
    assert any(branch.path == expected for branch in branches)
    assert not any(
        branch.path == (("a", "1.5") if outer else ("1.5",)) for branch in branches
    )


@pytest.mark.parametrize("child", ["i", "v", "x", "c"])
@pytest.mark.parametrize("separator", [" ", "\n"])
def test_roman_compatible_child_does_not_hoist_a_decimal_numbered_item(
    child, separator
):
    source = (
        "(1) Definitions.\n(2.5) (a) First:\n(I) First:\n(A) First:\n"
        "(1) The claimant must file a return.\n"
        "(1.5)" + separator + f"({child}) The claimant must submit an annual report.\n"
        "(3) The next subsection."
    )
    branches = _branches(source)
    assert any(branch.path == ("2.5", "a", "i", "a", "1.5") for branch in branches)
    assert not any(branch.path == ("1.5",) for branch in branches)
    assert any("must submit an annual report" in branch.text for branch in branches)


@pytest.mark.parametrize("child", ["a", "b"])
def test_returning_lower_alpha_sibling_does_not_hoist_a_decimal_numbered_item(child):
    source = (
        "(1) Definitions.\n(2.5) (a) First:\n(I) First:\n(A) First:\n"
        "(1) The claimant must file a return.\n"
        "(1.5) The claimant must submit an annual report.\n"
        f"({child}) An existing paragraph.\n(3) The next subsection."
    )
    branches = _branches(source)
    assert any(branch.path == ("2.5", "a", "i", "a", "1.5") for branch in branches)
    assert not any(branch.path == ("1.5",) for branch in branches)


@settings(max_examples=24, deadline=None)
@given(
    root=st.integers(min_value=1, max_value=8),
    inserted=st.lists(
        st.integers(min_value=1, max_value=9), min_size=1, max_size=4, unique=True
    ),
)
def test_inserted_decimal_roots_keep_their_own_children(root, inserted):
    labels = [
        str(root),
        *(f"{root}.{value}" for value in sorted(inserted)),
        str(root + 1),
    ]
    source = "".join(
        f"({label}) (a) The department must provide information for this subsection.\n"
        for label in labels
    )
    assert [branch.path for branch in _branches(source)] == [
        path for label in labels for path in [(label,), (label, "a")]
    ]


def test_decimal_markers_are_stripped_without_losing_the_source_body():
    text = "(1.5)(b) The amount of interest income from state and local bonds."
    assert sc._strip_source_clause_marker(text) == (
        "The amount of interest income from state and local bonds."
    )


def test_invalid_attached_label_does_not_prove_a_partial_numeric_outline():
    source = (
        "(2) Prior rule.\n(2)(iiv) Editorial prose.\n"
        "(3) Current rule.\n(i) Independent detail."
    )
    assert [branch.path for branch in _branches(source)] == [("2",), ("3",), ("i",)]


@pytest.mark.parametrize(
    "citation",
    [
        "us-ar/manual/dhs/snap-july-2026-final-filing/page-73",
        "us-ar/guidance/dhs/child-support/page-73",
        "us-ar/statute/20/502-914",
    ],
)
def test_partial_uppercase_outline_cannot_reparent_a_following_topic(citation):
    # Arkansas's page 72 introduces subsection (a), whose children (3) and
    # (4) continue on page 73 before the sibling subsection (b) begins.
    source = """\
(3)(A) Child support sanctions will be processed automatically through ANSWER.
(B) ANSWER will process sanction request files from the Office of Child Support Enforcement each night, then complete a budget to apply the sanction.
(4)(A) However, the payment reduction is the only appealable issue to the Department of Human Services.
(B) Since the Office of Child Support Enforcement made the noncompliance decision and has already provided the client an opportunity to appeal it, the noncompliance decision is not an appealable issue with the Department of Human Services.
(b) Lifting the child support sanction.
(1) A child support sanction may be lifted at any time a parent or other adult caretaker relative complies with the Office of Child Support Enforcement.
(2) If the parent or other adult caretaker relative wishes to have the sanction lifted by complying with the Office of Child Support Enforcement:
(A) A referral will be made to the Office of Child Support Enforcement; and
(B) Notification from the Office of Child Support Enforcement that he or she has cooperated must be received prior to the assistance being restored to the full amount.
"""
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    heading = next(
        branch
        for branch in branches
        if "Lifting the child support sanction" in branch.text
    )
    assert heading.path == ("b",)
    assert next(
        branch for branch in branches if branch.text.startswith("(1)")
    ).path == ("1",)
    assert next(
        branch for branch in branches if branch.text.startswith("(2)")
    ).path == ("2",)
    assert not any(branch.path[:1] == ("4",) for branch in branches)


@pytest.mark.parametrize(
    "citation", ["", "us-co/manual/dor/refunds/page-6", "us-co/guidance/dor/refunds"]
)
def test_partial_numeric_outline_requires_statutory_context(citation):
    source = (
        "(4) (a) The executive director shall adjust:\n"
        "(I) The amount of adjusted gross income.\n"
        "(b) Use the most recent estimate.\n(5) The next subsection."
    )
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    assert [branch.path for branch in branches] == [("4",), ("i",), ("b",), ("5",)]


@settings(max_examples=24, deadline=None)
@given(
    root=st.integers(min_value=2, max_value=98),
    citation=st.sampled_from(
        ["", "us-co/manual/dor/refunds/page-6", "us-co/guidance/dor/refunds"]
    ),
)
def test_incomplete_nonstatutory_pages_retain_existing_owner_paths(root, citation):
    source = (
        f"({root}) (a) The executive director shall adjust:\n"
        "(I) The amount of adjusted gross income.\n"
        f"(b) Use the most recent estimate.\n({root + 1}) The next subsection."
    )
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    assert [branch.path for branch in branches] == [
        (str(root),),
        ("i",),
        ("b",),
        (str(root + 1),),
    ]


@pytest.mark.parametrize(
    "prefix",
    [
        "The department shall review the preceding subsection.\n",
        "The following subsections:\n",
        "The department calls this a subsection\n\n",
        "For purposes of this subsection\n",
    ],
)
def test_reference_words_do_not_hide_real_subsection_content(prefix):
    source = "(1) " + prefix + "(2) The department must provide requested information."
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {("1",), ("2",)}
    assert "must provide requested information" in branches[("2",)].text


@pytest.mark.parametrize("boundary", [".\n", "\n\n", "\r\n\r\n", "\n \n"])
def test_reference_tokens_do_not_cross_sentence_or_blank_paragraph_boundaries(boundary):
    source = (
        "(1) Refunds follow this subsection"
        + boundary
        + "(2) Of this section's refunds, only residents may claim a refund."
    )
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {("1",), ("2",)}
    assert "only residents may claim" in branches[("2",)].text


@pytest.mark.parametrize(
    "intro",
    [
        "The following subsections",
        "See the following paragraphs. The following subsections",
        "The following definitions in this subsection",
        "For purposes of this subsection",
    ],
)
@pytest.mark.parametrize("separator", [" ", "\n", "\r\n", "\n  "])
def test_unpunctuated_following_subsections_chapeau_keeps_joined_repeal_roots(
    intro, separator
):
    source = (
        "(1) " + intro + "\n(2) and" + separator + "(3) Repealed.\n"
        "(4) The claimant must file a return."
    )
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {("1",), ("2",), ("3",), ("4",)}
    assert sc._source_branch_is_explicit_repeal_tombstone(branches[("2",)].text)
    assert sc._source_branch_is_explicit_repeal_tombstone(branches[("3",)].text)


def test_explicit_following_subsections_reference_stays_in_its_owning_branch():
    source = WRAPPED_REFERENCE.replace(
        "provided in subsection\n(3)",
        "provided in the following subsections\n(3) or (5)(a)",
    )
    assert [branch.path for branch in _branches(source)] == [
        ("4",),
        ("4", "a"),
        ("4", "a", "i"),
        ("4", "b"),
        ("5",),
    ]


def test_joined_repeal_covers_every_named_branch_with_the_same_source_span():
    branches = {branch.path: branch for branch in _branches(COMBINED_REPEAL)}
    assert branches[("1", "b", "ii")].label == "(II)"
    assert branches[("1", "b", "iii")].label == "(III)"
    for first, second in [(("1", "b", "ii"), ("1", "b", "iii")), (("8",), ("9",))]:
        assert first in branches
        assert second in branches
        assert branches[first].text == branches[second].text
        assert branches[first].start == branches[second].start
        assert branches[first].end == branches[second].end
        assert sc._reason_covers_source_authenticated_repeal(
            CITATION
            + "".join(f"({label})" for label in second)
            + " is repealed and supplies no operative rule.",
            branches[second].text,
            path=second,
            corpus_citation_path=CITATION,
        )


def test_joined_repeal_does_not_swallow_the_following_decimal_subsection():
    source = (
        "(1) First.\n(8) and (9) Repealed.\n"
        "(9.5) Section 39-21-108 (3) applies to the refund.\n(10) Last."
    )
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {("1",), ("8",), ("9",), ("9.5",), ("10",)}
    assert branches[("8",)].text == "(8) and (9) Repealed."
    assert branches[("9",)].text == "(8) and (9) Repealed."


def test_wrapped_joined_repeals_keep_complete_authenticated_source_spans():
    source = COMBINED_REPEAL.replace(" and (", " and\n(")
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {branch.path for branch in _branches(COMBINED_REPEAL)}
    for first, second in [(("1", "b", "ii"), ("1", "b", "iii")), (("8",), ("9",))]:
        assert branches[first].text == branches[second].text
        assert branches[first].start == branches[second].start
        assert branches[first].end == branches[second].end
        for path in (first, second):
            assert sc._reason_covers_source_authenticated_repeal(
                CITATION
                + "".join(f"({label})" for label in path)
                + " is repealed and supplies no operative rule.",
                branches[path].text,
                path=path,
                corpus_citation_path=CITATION,
            )


def test_wrapped_repeal_union_keeps_the_next_indented_decimal_branch_separate():
    source = (
        "(8) and\n(9) Repealed.\n  (9.5) The claimant must file a return.\n\t(10) Last."
    )
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {("8",), ("9",), ("9.5",), ("10",)}
    assert branches[("8",)].text == branches[("9",)].text == "(8) and\n(9) Repealed."
    assert "must file a return" in branches[("9.5",)].text


@pytest.mark.parametrize(
    "continuation",
    [" Taxpayers must file a return.", "\nTaxpayers must file a return."],
)
def test_wrapped_joined_repeal_does_not_merge_or_defer_operative_continuation(
    continuation,
):
    source = "(1) First.\n(8) and\n(9) Repealed." + continuation + "\n(10) Last."
    branches = {branch.path: branch for branch in _branches(source)}
    assert branches[("8",)].end == branches[("9",)].start
    assert "Taxpayers must file a return" in branches[("9",)].text
    for path in [("8",), ("9",)]:
        assert not sc._reason_covers_source_authenticated_repeal(
            CITATION
            + "".join(f"({label})" for label in path)
            + " is repealed and supplies no operative rule.",
            branches[path].text,
            path=path,
            corpus_citation_path=CITATION,
        )


def test_exact_joined_repeal_deferrals_pass_the_complete_source_coverage_check():
    paths = [("1", "b", "ii"), ("1", "b", "iii"), ("8",), ("9",)]
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "us-co:statutes/39/39-22-2003/"
                    + "/".join(path)
                    + "#repealed_subsection",
                    "reason": CITATION
                    + "".join(f"({label})" for label in path)
                    + " is repealed and supplies no operative rule.",
                }
                for path in paths
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload,
        source_text=COMBINED_REPEAL,
        branches=_branches(COMBINED_REPEAL),
        corpus_citation_path=CITATION,
    )
    assert issues == []
    assert covered == set(paths)


@settings(max_examples=24, deadline=None)
@given(
    peer_count=st.integers(min_value=1, max_value=48),
    breaks=st.sets(st.integers(min_value=3, max_value=50), max_size=20),
)
def test_finite_joined_repeal_list_preserves_every_named_source_branch(
    peer_count, breaks
):
    peers = list(range(3, 3 + peer_count))
    locator = "(2)" + "".join(
        "," + ("\n" if label in breaks else " ") + f"({label})" for label in peers[:-1]
    )
    locator += ", and" if len(peers) > 1 else " and"
    locator += ("\n" if peers[-1] in breaks else " ") + f"({peers[-1]})"
    source = "(1) First.\n" + locator + " Repealed.\n(100) Last."
    branches = {branch.path: branch for branch in _branches(source)}
    assert set(branches) == {
        ("1",),
        ("100",),
        ("2",),
        *((str(label),) for label in peers),
    }
    for label in (2, *peers):
        branch = branches[(str(label),)]
        assert branch.text == locator + " Repealed."
        assert source[branch.start : branch.end].strip() == branch.text
        assert sc._source_branch_is_explicit_repeal_tombstone(branch.text)


def test_ordinary_comma_parentheses_do_not_create_new_source_markers():
    source = (
        "(1) The comparison cites:\n"
        "(2), (3), and (4) of this section as supporting references.\n"
        "(5) The department must provide requested information."
    )
    assert [branch.path for branch in _branches(source)] == [("1",), ("5",)]


@settings(max_examples=24, deadline=None)
@given(first=st.integers(min_value=2, max_value=98))
def test_joined_repeal_authentication_requires_each_exact_named_branch(first):
    source = f"(1) First.\n({first}) and ({first + 1}) Repealed.\n(100) Last."
    branches = {branch.path: branch for branch in _branches(source)}
    for label in (str(first), str(first + 1)):
        branch = branches[(label,)]
        reason = CITATION + f"({label}) is repealed and supplies no operative rule."
        assert sc._reason_covers_source_authenticated_repeal(
            reason,
            branch.text,
            path=branch.path,
            corpus_citation_path=CITATION,
        )
        assert not sc._reason_covers_source_authenticated_repeal(
            reason.replace(f"({label})", "(101)"),
            branch.text,
            path=branch.path,
            corpus_citation_path=CITATION,
        )
        assert not sc._reason_covers_source_authenticated_repeal(
            reason,
            branch.text + " Taxpayers must file a return.",
            path=branch.path,
            corpus_citation_path=CITATION,
        )


@settings(max_examples=32, deadline=None)
@given(
    lead=st.sampled_from([["subsection"], ["section", "39-22-2002"]]),
    locator=st.sampled_from(
        [
            ["(3)"],
            ["(2)", "or", "(5)", "(a)"],
            ["(2)", "to", "(5)", "(a)"],
            ["(2)", "through", "(5)", "(a)"],
            ["(1.5)", "(b)"],
        ]
    ),
    breaks=st.sets(st.integers(min_value=0, max_value=6), max_size=6),
    separator=st.sampled_from([" ", "\n", "\n  ", "\r\n"]),
)
def test_reference_line_wrapping_preserves_legal_paths(
    lead, locator, breaks, separator
):
    tokens = [*lead, *locator]
    reference = "".join(
        (separator if index in breaks else " ") + token
        for index, token in enumerate(tokens)
    )
    source = (
        "(4) (a) The refund is as provided in"
        + reference
        + " of this section; the amount shall be adjusted:\n"
        "(I) Use adjusted gross income.\n(b) Use the most recent estimate.\n"
        "(5) The next subsection.\n"
    )
    unwrapped = source.replace(reference, " " + " ".join(tokens), 1)
    expected = [("4",), ("4", "a"), ("4", "a", "i"), ("4", "b"), ("5",)]
    assert [branch.path for branch in _branches(source)] == expected
    assert [branch.path for branch in _branches(unwrapped)] == expected


@settings(max_examples=32, deadline=None)
@given(
    root=st.integers(min_value=2, max_value=98),
    decimal=st.integers(min_value=1, max_value=9),
    breaks=st.sets(st.integers(min_value=0, max_value=5), max_size=6),
    separator=st.sampled_from([" ", "\n", "\n  ", "\r\n"]),
)
def test_decimal_self_reference_wrapping_preserves_legal_paths(
    root, decimal, breaks, separator
):
    label = f"{root}.{decimal}"
    tokens = ["this", "subsection", f"({label})", "for", "each", "lease"]
    reference = "".join(
        (separator if index in breaks else " ") + token
        for index, token in enumerate(tokens)
    )
    source = (
        "(1) Definitions.\n"
        f"({label}) (a) The credit amount is set forth in paragraph (b) of"
        + reference
        + " of a truck during the tax year.\n"
        "(b) The credit depends on the purchase year.\n"
        f"({root + 1}) The next subsection."
    )
    unwrapped = source.replace(reference, " " + " ".join(tokens), 1)
    expected = [("1",), (label,), (label, "a"), (label, "b"), (str(root + 1),)]
    assert [branch.path for branch in _branches(source)] == expected
    assert [branch.path for branch in _branches(unwrapped)] == expected


@pytest.mark.parametrize(
    "tail",
    [
        "Repealed; taxpayers must file a return.",
        "Repealed. Taxpayers must file a return.",
        "Repeal proposed.",
        "Not repealed.",
    ],
)
def test_joined_locator_does_not_authenticate_operative_or_proposed_repeal(tail):
    source = f"(1) First.\n(8) and (9) {tail}\n(10) Last."
    assert not sc._source_branch_is_explicit_repeal_tombstone(f"(8) and (9) {tail}")
    for branch in _branches(source):
        if branch.path == ("9",):
            assert not sc._reason_covers_source_authenticated_repeal(
                CITATION + "(9) is repealed and supplies no operative rule.",
                branch.text,
                path=branch.path,
                corpus_citation_path=CITATION,
            )
