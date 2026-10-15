"""Colorado source paths survive PDF wrapping and shared repeal locators."""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)

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


def _inserted_alpha_source(labels):
    return (
        "(a) Earlier rule.\n(b) Earlier rule.\n"
        + "".join(
            f"({label}) Property classes:\n"
            "(1) First class.\n(2) Second class:\n"
            "(A) Improved residential real property which:\n"
            "(i) Is not occupied by the owner.\n"
            "(ii) Contains not more than 5 dwelling units.\n"
            "(B) Another class.\n(3) Third class.\n"
            for label in labels
        )
        + "(d) The next subsection."
    )


def test_inserted_alpha_roots_preserve_each_original_threshold_demand():
    source = _inserted_alpha_source(["c", "c-1", "c-2"])
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us-dc/statute/47/47-813"
    )
    paths = [branch.path for branch in branches]
    assert len(paths) == len(set(paths))
    for label in ("c", "c-1", "c-2"):
        assert (label, "2", "a", "ii") in paths
    boundaries = sc._source_boundary_obligations(
        branches,
        extract_numeric_occurrences=lambda text: (
            extract_typed_numeric_inventory_occurrences_from_text(text, profile="en-US")
        ),
    )
    five_unit = [
        (branch, numeric)
        for branch, numeric in boundaries
        if numeric.value == 5 and numeric.raw == "5"
    ]
    assert {branch.path for branch, _ in five_unit} == {
        (label, "2", "a", "ii") for label in ("c", "c-1", "c-2")
    }
    for branch, numeric in five_unit:
        recall = sc.authoritative_numeric_recall_text(
            sc._source_branch_direct_text(branch, branches=branches)
        )
        assert recall[numeric.start : numeric.end] == "5"
        assert branch.text == "(ii) Contains not more than 5 dwelling units."


def test_inline_serial_roman_clause_keeps_its_uppercase_children():
    source = (
        "(A) The credit is allowed only if— (i) the expenditures benefit dwelling "
        "units, and (ii) the amount during any 24-month period meets the greater "
        "of the following subclauses: (I) The amount is at least 20 percent of "
        "the basis. (II) The amount is at least $6,000. "
        "(B) The requirement of subparagraph (A)(ii)(I) may be adjusted. "
        "(C) The requirement of subparagraph (A)(ii)(II) may be adjusted."
    )
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us/statute/26/42/e/3"
    )
    paths = {branch.path for branch in branches}
    assert {("a", "i"), ("a", "ii"), ("a", "ii", "i"), ("a", "ii", "ii")} <= paths
    first = next(branch for branch in branches if branch.path == ("a", "i"))
    second = next(branch for branch in branches if branch.path == ("a", "ii"))
    assert first.end == second.start
    assert "24-month" not in first.text
    assert "20 percent" in second.text and "$6,000" in second.text


@pytest.mark.parametrize(
    "reference",
    [
        "under paragraphs (i) and (ii) of this section",
        "under (i) and (ii) of this section",
        "pursuant to (i) and (ii) of this section",
        "as provided in (i) and (ii) of this section",
        "see (i), and (ii) of this section",
        "as noted, see (i) and (ii) of this section",
        'the quoted words "see (i), and (ii)"',
    ],
)
def test_serial_inline_roman_reference_cannot_become_a_clause(reference):
    source = (
        "(A) The credit is allowed only if— (i) the amount is prescribed "
        + reference
        + " for the following subclauses:\n(I) Use the credit amount.\n"
        "(II) Use the income amount.\n(B) The next provision."
    )
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us/statute/26/42/e/3"
    )
    assert not any(branch.path == ("a", "ii") for branch in branches)


@pytest.mark.parametrize(
    "reference", ["subsection\n(c-1)", "subsections (c-1) to\n(c-2)"]
)
def test_wrapped_inserted_alpha_references_keep_their_owner(reference):
    source = (
        "(a) Earlier rule.\n(b) As provided in "
        + reference
        + " of this section.\n(c) Next rule."
    )
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us-dc/statute/47/47-813"
    )
    assert [branch.path for branch in branches] == [("a",), ("b",), ("c",)]


def test_inserted_alpha_self_reference_for_keeps_only_the_current_root():
    source = (
        "(a) Earlier rule.\n(b) Earlier rule.\n(c) Earlier rule.\n"
        "(c-1) The credit is allowed under this subsection\n(c-1) for each lease.\n"
        "(c-2) For purposes of this subsection, the claimant must file a return.\n(d) Next rule."
    )
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us-dc/statute/47/47-813"
    )
    assert [branch.path for branch in branches] == [
        ("a",),
        ("b",),
        ("c",),
        ("c-1",),
        ("c-2",),
        ("d",),
    ]


def test_inserted_alpha_repeal_binds_only_its_exact_canonical_branch():
    source = "(a) First.\n(b) First.\n(c) First.\n(c-5) Repealed.\n(d) Last."
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us-dc/statute/47/47-813"
    )
    branch = next(branch for branch in branches if branch.path == ("c-5",))
    assert sc._strip_source_clause_marker(branch.text) == "Repealed."
    assert sc._reason_covers_source_authenticated_repeal(
        "us-dc/statute/47/47-813(c-5) is repealed.",
        branch.text,
        path=branch.path,
        corpus_citation_path="us-dc/statute/47/47-813",
    )
    assert not sc._reason_covers_source_authenticated_repeal(
        "us-dc/statute/47/47-813(c-4) is repealed.",
        branch.text,
        path=branch.path,
        corpus_citation_path="us-dc/statute/47/47-813",
    )


@pytest.mark.parametrize("label", ["c-0", "c-01", "c-word", "c-1-2", "1-2"])
def test_inserted_alpha_labels_require_one_positive_integer_suffix(label):
    assert not sc._is_parenthesized_outline_label(label)


@settings(max_examples=24, deadline=None)
@given(separator=st.sampled_from([" ", "\n", "\r\n", "\n  "]))
def test_serial_roman_clause_wrapping_preserves_its_child_owners(separator):
    source = (
        "(A) The credit is allowed only if— (i) the expenditures benefit dwelling units, and"
        + separator
        + "(ii) the amount meets the requirements of the following subclauses:"
        + separator
        + "(I) The amount is at least 20 percent of the basis. "
        "(II) The amount is at least $6,000. (B) The next provision."
    )
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us/statute/26/42/e/3"
    )
    assert [(branch.path) for branch in branches] == [
        ("a",),
        ("a", "i"),
        ("a", "ii"),
        ("a", "ii", "i"),
        ("a", "ii", "ii"),
        ("b",),
    ]


@pytest.mark.parametrize("returning", ["iii", "iv", "v"])
def test_deep_attached_roman_child_keeps_every_explicit_ancestor(returning):
    source = (
        "(1) Earlier rule.\n(2)(a)(I)(A)(1)(i)(A)(1)(ii)(I) The amount is at least 40 percent.\n"
        "(II) The amount is at least 50 percent.\n"
        f"(III) The amount is at least 60 percent.\n({returning}) The next parent clause.\n"
        "(3) The next subsection."
    )
    branches = _branches(source)
    expected_parent = ("2", "a", "i", "a", "1", "i", "a", "1", "ii")
    assert any(branch.path == (*expected_parent, "i") for branch in branches)
    assert any(branch.path == (*expected_parent, "ii") for branch in branches)
    assert any(branch.path == (*expected_parent, "iii") for branch in branches)
    assert any(branch.path == (*expected_parent[:-1], returning) for branch in branches)


@settings(max_examples=20, deadline=None)
@given(depth=st.integers(min_value=1, max_value=5))
def test_attached_legal_chains_preserve_ancestors_beyond_the_conventional_profile(
    depth,
):
    labels = ["2", "a", *("I", "A", "1") * depth, "ii", "I"]
    source = (
        "(1) Earlier rule.\n"
        + "".join(f"({label})" for label in labels)
        + " The amount is at least 40 percent.\n(3) The next subsection."
    )
    assert any(
        branch.path == tuple(label.lower() for label in labels)
        for branch in _branches(source)
    )


@settings(max_examples=20, deadline=None)
@given(copies=st.integers(min_value=2, max_value=5))
def test_distinct_original_boundary_copies_survive_exact_duplicate_extraction(copies):
    text = "The total decrease shall not exceed 40 percent of federal taxable income."
    branches = [
        sc.SourceStructureBranch(
            ("21", "b", "ii", "i"),
            "formula-clause",
            "(I)",
            text,
            index * (len(text) + 1),
            index * (len(text) + 1) + len(text),
        )
        for index in range(copies)
    ]
    obligations = sc._source_boundary_obligations(
        (),
        narrative_formula_branches=(*branches, *branches),
        extract_numeric_occurrences=lambda value: (
            extract_typed_numeric_inventory_occurrences_from_text(
                value, profile="en-US"
            )
        ),
    )
    assert len(obligations) == copies
    assert {branch.start for branch, numeric in obligations if numeric.value == 40} == {
        branch.start for branch in branches
    }


@settings(max_examples=24, deadline=None)
@given(
    suffixes=st.integers(min_value=1, max_value=5),
    whitespace=st.sampled_from(["\n", "\r\n", "\n\n"]),
)
def test_inserted_alpha_roots_keep_repeated_nested_locators_distinct(
    suffixes, whitespace
):
    labels = ["c", *(f"c-{index}" for index in range(1, suffixes + 1))]
    source = _inserted_alpha_source(labels).replace("\n", whitespace)
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us-dc/statute/47/47-813"
    )
    paths = [branch.path for branch in branches]
    assert len(paths) == len(set(paths))
    for label in labels:
        assert (label, "2", "a", "ii") in paths
        branch = next(
            branch for branch in branches if branch.path == (label, "2", "a", "ii")
        )
        assert branch.text == "(ii) Contains not more than 5 dwelling units."


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


@pytest.mark.parametrize("root", ["1", "11.5"])
def test_uppercase_roman_returns_to_its_parent_after_uppercase_letter_children(root):
    # C.R.S. 39-22-303(11.5)(b)(IV) follows (III)(A) through (D).
    source = (
        ("(1) Prior rule.\n" if root != "1" else "")
        + f"({root}) (b) For tax years beginning on and after January 1, 2026:\n"
        "(I) The members shall file a combined report.\n"
        "(II) The net income of each member is combined.\n"
        "(III) (A) Income or loss is allocated and apportioned to this state.\n"
        "(B) The combined group apportionment factor is a fraction.\n"
        "(C) Intercompany transactions are excluded.\n"
        "(D) Partnership factors are included in the combined group's factors.\n"
        "(IV) The combined report must be filed under the name of the parent.\n"
        "(V) Members of the combined group are jointly and severally liable.\n"
        "(VI) The executive director shall not require consolidated returns.\n"
        f"({int(root.split('.')[0]) + 1}) The next subsection."
    )
    branches = _branches(source)
    expected = ([("1",)] if root != "1" else []) + [
        (root,),
        (root, "b"),
        (root, "b", "i"),
        (root, "b", "ii"),
        (root, "b", "iii"),
        *((root, "b", "iii", child) for child in "abcd"),
        (root, "b", "iv"),
        (root, "b", "v"),
        (root, "b", "vi"),
        (str(int(root.split(".")[0]) + 1),),
    ]
    assert [branch.path for branch in branches] == expected
    third = next(branch for branch in branches if branch.path == (root, "b", "iii"))
    fourth = next(branch for branch in branches if branch.path == (root, "b", "iv"))
    assert third.end == fourth.start
    assert "(D) Partnership" in third.text
    assert "(IV)" not in third.text


_UPPERCASE_ROMAN_SEQUENCE = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
)


@settings(max_examples=40, deadline=None)
@given(
    root=st.sampled_from(["1", "2.5", "11.5"]),
    length=st.integers(min_value=2, max_value=len(_UPPERCASE_ROMAN_SEQUENCE)),
    child_count=st.integers(min_value=1, max_value=9),
    attached=st.booleans(),
    blank_lines=st.integers(min_value=0, max_value=2),
)
def test_sequential_uppercase_roman_siblings_keep_intervening_letter_children(
    root, length, child_count, attached, blank_lines
):
    children = "ABCDEFGHI"[:child_count]
    source = ("(1) Prior rule.\n" if root != "1" else "") + f"({root}) (b) Rules:\n"
    expected = ([("1",)] if root != "1" else []) + [(root,), (root, "b")]
    for label in _UPPERCASE_ROMAN_SEQUENCE[:length]:
        source += f"({label})" + (" " if attached else " The provision applies:\n")
        expected.append((root, "b", label.lower()))
        for child in children:
            source += f"({child}) The department must provide information.\n"
            expected.append((root, "b", label.lower(), child.lower()))
    next_root = str(int(root.split(".")[0]) + 1)
    source += f"({next_root}) The next subsection."
    expected.append((next_root,))
    source = source.replace("\n", "\n" * (blank_lines + 1))
    assert [branch.path for branch in _branches(source)] == expected


@pytest.mark.parametrize("layout", ["i_to_ii", "i_to_a", "all_markers"])
def test_blank_paragraphs_preserve_proven_uppercase_roman_owners(layout):
    source = (
        "(1) Prior rule.\n(11.5) (b) Rules:\n(I) The first provision applies.\n"
        "(II) The second provision applies.\n"
        "(III) (A) Income is apportioned.\n(B) The factor applies.\n"
        "(C) Transactions are excluded.\n(D) Partnership factors apply.\n"
        "(IV) The report must be filed.\n(V) Members are liable.\n"
        "(VI) Returns are required.\n(12) The next subsection."
    )
    if layout == "i_to_a":
        source = source.replace(
            "(I) The first provision applies.\n",
            "(I) The first provision applies:\n(A) The claimant must file a return.\n",
        )
        changed = source.replace("applies:\n(A)", "applies:\n\n(A)")
    elif layout == "i_to_ii":
        changed = source.replace("applies.\n(II)", "applies.\n\n(II)", 1)
    else:
        changed = source.replace("\n", "\n\n")
    expected = [branch.path for branch in _branches(source)]
    actual = [branch.path for branch in _branches(changed)]
    assert ("11.5", "b", "iv") in expected
    assert actual == expected


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("I", "II", False),
        ("III", "IV", False),
        ("IV", "V", False),
        ("IX", "X", False),
        ("X", "XI", False),
        ("A", "B", True),
        ("C", "D", True),
        ("H", "I", True),
        ("I", "J", True),
        ("I", "V", False),
        ("III", "V", False),
        ("D", "IV", False),
        ("IIV", "V", False),
        ("III", "IIV", False),
        ("ii", "III", False),
        ("II", "iii", False),
    ],
)
def test_uppercase_alphabet_order_is_preserved_for_roman_compatible_letters(
    previous, current, expected
):
    assert (
        sc._parenthesized_outline_label_follows(previous, current, "upper-alpha")
        is expected
    )


@pytest.mark.parametrize("label", ["I", "II", "III", "IV", "V", "IX", "X", "C", "D"])
def test_uppercase_valid_roman_labels_keep_both_outline_categories(label):
    assert sc._parenthesized_legal_outline_categories(label) == ("upper-alpha", "roman")


@pytest.mark.parametrize("label", ["A", "B", "E", "IIV", "IIX", "VX"])
def test_uppercase_nonroman_labels_cannot_prove_a_roman_continuation(label):
    assert sc._parenthesized_legal_outline_categories(label) == ("upper-alpha",)


def test_uppercase_letter_siblings_keep_ambiguous_roman_letters_at_their_own_level():
    source = (
        "(1) (b) Rules:\n"
        + "".join(
            f"({label}) The department must provide information.\n"
            for label in "ABCDEFGHIJK"
        )
        + "(2) The next subsection."
    )
    assert [branch.path for branch in _branches(source)] == [
        ("1",),
        ("1", "b"),
        *(("1", "b", label.lower()) for label in "ABCDEFGHIJK"),
        ("2",),
    ]


@pytest.mark.parametrize("child_count", [0, 1])
@pytest.mark.parametrize("attached", [False, True])
def test_partial_uppercase_alphabet_keeps_i_j_k_as_siblings(child_count, attached):
    prefix = (
        "(11.5) (b) (I) The first provision applies:\n"
        if attached
        else "(11.5) (b) Rules:\n(I) The first provision applies:\n"
    )
    source = "(1) Prior rule.\n" + prefix
    expected = [("1",), ("11.5",), ("11.5", "b"), ("11.5", "b", "i")]
    if child_count:
        source += "(1) The department must provide information.\n"
        expected.append(("11.5", "b", "i", "1"))
    source += "(J) The next provision applies.\n(K) The last provision applies.\n(12) The next subsection."
    expected.extend([("11.5", "b", "j"), ("11.5", "b", "k"), ("12",)])
    assert [branch.path for branch in _branches(source)] == expected


@settings(max_examples=30, deadline=None)
@given(
    initial=st.sampled_from(["C", "D", "I", "L", "M", "V", "X"]),
    child_count=st.integers(min_value=0, max_value=4),
    attached=st.booleans(),
)
def test_partial_alphabet_successors_prove_ambiguous_uppercase_letter_ownership(
    initial, child_count, attached
):
    source = "(1) Prior rule.\n(11.5) (b)" + (" " if attached else " Rules:\n")
    expected = [("1",), ("11.5",), ("11.5", "b")]
    for index in range(3):
        label = chr(ord(initial) + index)
        source += f"({label}) The provision applies:\n"
        expected.append(("11.5", "b", label.lower()))
        for number in range(1, child_count + 1):
            source += f"({number}) The department must provide information.\n"
            expected.append(("11.5", "b", label.lower(), str(number)))
    source += "(12) The next subsection."
    expected.append(("12",))
    assert [branch.path for branch in _branches(source)] == expected


@pytest.mark.parametrize("root", ["1", "11.5"])
def test_a_b_under_a_numeric_child_do_not_prove_the_parent_i_is_roman(root):
    source = (
        ("(1) Prior rule.\n" if root != "1" else "")
        + f"({root}) (b) Rules:\n(I) The first provision applies:\n"
        "(1) The rule applies:\n(A) The department must provide information.\n"
        "(B) The claimant must file a return.\n"
        "(J) The next provision applies.\n(K) The last provision applies.\n"
        f"({int(root.split('.')[0]) + 1}) The next subsection."
    )
    expected = ([("1",)] if root != "1" else []) + [
        (root,),
        (root, "b"),
        (root, "b", "i"),
        (root, "b", "i", "1"),
        (root, "b", "i", "1", "a"),
        (root, "b", "i", "1", "b"),
        (root, "b", "j"),
        (root, "b", "k"),
        (str(int(root.split(".")[0]) + 1),),
    ]
    assert [branch.path for branch in _branches(source)] == expected


def test_wrapped_reference_cannot_prove_a_roman_child_in_the_prior_stack():
    source = (
        "(1) Prior rule.\n(11.5) (b) Rules:\n"
        "(I) The amount is provided in paragraph\n(A) of this section.\n"
        "(J) The credit is provided in this subsection\n(11.5) for each lease.\n"
        "(K) The last provision applies.\n(12) The next subsection."
    )
    matches = tuple(sc._PARENTHESIZED_OUTLINE_MARKER.finditer(source))
    index = next(
        index for index, match in enumerate(matches) if match.group("marker") == "(I)"
    )
    assert (
        sc._parenthesized_outline_following_sibling_label(
            source, matches, index, _active_numeric_root="11.5"
        )
        == "J"
    )
    branches = _branches(source)
    assert [branch.path for branch in branches] == [
        ("1",),
        ("11.5",),
        ("11.5", "b"),
        ("11.5", "b", "i"),
        ("11.5", "b", "j"),
        ("11.5", "b", "k"),
        ("12",),
    ]
    assert "(A) of this section" in branches[3].text
    assert "(11.5) for each lease" in branches[4].text


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


# Exact pinned Arkansas manual bodies; arithmetic clauses must keep their owners.
AR_PARTIAL_INITIAL = "DRAFT 7 09/05/2025 12:28:25 PM\n\n(D) Refer to 20 CAR §§ 502-504 and 502-510 for unearned and earned income to disregard.\n\n(2)(A) The Program Eligibility Specialist will deny the application if the participant fails to meet the income requirement upon receipt of initial work documentation.\n\n(B) A TEA-1 will be sent as an adequate notice of the application denial.\n\n(C) See 20 CAR §§ 503-302 and 503-303 for more information on adequate notices.\n\n(3) To determine if the household meets the federal poverty level requirement for initial eligibility, the Program Eligibility Specialist will use one (1) of the appropriate calculations indicated below.\n\n(4)(A) If the participant is paid weekly:\n\n(i) Total the gross income actually received in the thirty-day period;\n\n(ii)(a) Divide the total by four (4).\n\n(b) See note below; and\n\n(iii) Multiply by 4.334.\n\n(B) Note. If the participant is paid weekly but received five (5) checks in the thirty-day period, divide total gross income by five (5) and multiply by 4.334.\n\n(5)(A) If the participant is paid every two (2) weeks (biweekly):\n\n(i) Total the gross income actually received in the thirty-day period;\n\n(ii)(a) Divide the total by two (2).\n\n(b) See note below; and\n\n(iii) Multiply by 2.167.\n\n(B) Note. If the participant is paid every two (2) weeks (biweekly) but received three (3) checks in the thirty-day period, divide total gross income by three (3) and multiply by 2.167.\n\n(6) If the participant is paid twice a month (semi-monthly), total the gross income actually received in the thirty-day period, with no conversion."
AR_PARTIAL_ONGOING = "DRAFT 24 09/05/2025 12:28:25 PM\n\n(B) See Appendix B.\n\n(2) The income of the child or children is disregarded.\n\n(3) Unemployment insurance and child support are also disregarded when determining initial (and ongoing) eligibility for the Arkansas Work Pays Program.\n\n(4) Refer to 20 CAR § 502-506 for unearned income to disregard.\n\n(f) FPL calculation (ongoing eligibility).\n\n(1)(A) If the participant is paid weekly:\n\n(i) Total the gross income actually received in the month;\n\n(ii)(a) Divide the total by four (4).\n\n(b) See note below; and\n\n(iii) Multiply by 4.334.\n\n(B) Note. If the participant is paid weekly but received five (5) checks in the month, divide total gross income by and multiply by 4.334.\n\n(2)(A) If the participant is paid every two (2) weeks (biweekly):\n\n(i) Total the gross income actually received in the month;\n\n(ii)(a) Divide the total by two (2).\n\n(b) See note below; and\n\n(iii) Multiply by 2.167.\n\n(B) Note. If the participant is paid every two (2) weeks (biweekly) but received three (3) checks in the month, divide total gross income by three (3) and multiply by 2.167.\n\n(3) If the participant is paid twice a month (semi-monthly), total the gross income actually received in the month with no conversion.\n\n(4) Note. The Program Eligibility Specialist will review the work documentation each month to determine if there is a need to update the budget.\n\n(5)(A) An update to the budget is only necessary if there has been a significant change in income.\n\n(B) A significant change is:\n\n(i) One in which the new wage is equal to or exceeds one hundred fifty percent (150%) of the current federal poverty level for the family size; and"


@pytest.mark.parametrize(
    ("source", "citation", "expected_paths"),
    [
        (
            AR_PARTIAL_INITIAL,
            "us-ar/manual/dhs/snap-july-2026-final-filing/page-84",
            [
                ("d", "2", "a"),
                ("d", "2", "b"),
                ("d", "4", "a", "ii", "a"),
                ("d", "4", "a", "ii", "b"),
                ("d", "4", "a", "iii"),
                ("d", "4", "b"),
                ("d", "5", "a", "ii", "a"),
                ("d", "5", "a", "ii", "b"),
                ("d", "5", "a", "iii"),
                ("d", "5", "b"),
            ],
        ),
        (
            AR_PARTIAL_ONGOING,
            "us-ar/manual/dhs/snap-july-2026-final-filing/page-101",
            [
                ("f", "1", "a", "ii", "a"),
                ("f", "1", "a", "ii", "b"),
                ("f", "1", "a", "iii"),
                ("f", "1", "b"),
                ("f", "2", "a", "ii", "a"),
                ("f", "2", "a", "ii", "b"),
                ("f", "2", "a", "iii"),
                ("f", "2", "b"),
                ("f", "5", "a"),
                ("f", "5", "b"),
                ("f", "5", "b", "i"),
            ],
        ),
    ],
    ids=["initial", "ongoing"],
)
def test_partial_manual_attached_extension_keeps_its_outline(
    source, citation, expected_paths
):
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    paths = [branch.path for branch in branches]
    assert all(path in paths for path in expected_paths)
    assert len(paths) == len(set(paths))
    by_path = {branch.path: branch for branch in branches}
    for path in expected_paths:
        branch = by_path[path]
        assert source[branch.start : branch.end].strip() == branch.text
        if path[-2:] == ("ii", "a"):
            assert "Divide the total" in sc._source_branch_direct_text(
                branch, branches=branches
            )
        elif path[-1] == "iii":
            assert "Multiply by" in sc._source_branch_direct_text(
                branch, branches=branches
            )


@pytest.mark.parametrize("label", ["b", "B"])
def test_dynamic_attached_category_overrides_its_nominal_profile_slot(label):
    active = [
        (0, "f", "f", "lower-alpha"),
        (1, "1", "1", "numeric"),
        (3, "A", "a", "upper-alpha"),
        (5, "ii", "ii", "roman"),
        (6, "a", "a", "lower-alpha"),
    ]
    assert sc._parenthesized_legal_outline_level(
        label,
        active=active,
        attached_parent_level=None,
        outline_levels=sc._PARENTHESIZED_LEGAL_OUTLINE_LEVELS,
    ) == ((6, "lower-alpha") if label == "b" else (3, "upper-alpha"))


@pytest.mark.parametrize("label", ["iii", "iv", "v"])
def test_dynamic_roman_parent_return_ignores_a_mismatched_nominal_slot(label):
    active = [
        (0, "f", "f", "lower-alpha"),
        (1, "1", "1", "numeric"),
        (3, "A", "a", "upper-alpha"),
        (5, "ii", "ii", "roman"),
        (6, "III", "iii", "roman"),
    ]
    assert sc._parenthesized_legal_outline_level(
        label,
        active=active,
        attached_parent_level=None,
        outline_levels=sc._PARENTHESIZED_LEGAL_OUTLINE_LEVELS,
    ) == (5, "roman")


@pytest.mark.parametrize(
    "ending",
    [
        "(c) Third condition.\n(b) Second condition.",
        "(2)(A) New computation.\n(b) Separate child.",
        "subclauses\n(b) of this section.",
    ],
)
def test_attached_extension_admission_needs_its_active_consecutive_sibling(ending):
    source = (
        "(f) Partial manual section.\n(1)(A) A monthly computation:\n"
        "(i) Total income.\n(ii)(a) Divide income by four.\n" + ending
    )
    assert not sc._qualified_parenthesized_legal_outline_markers(
        source, corpus_citation_path="us-ar/manual/example"
    )


@pytest.mark.parametrize(
    "source",
    [
        "(f) The rule refers to subparagraphs\n"
        "(1)(A)(i)(A)(1)(i)(A)(1)(ii)(a) and\n(b) of this section.",
        "(f) Partial manual section.\n(2)(A) First condition.\n(B) Second condition.",
    ],
)
def test_attached_extension_admission_preserves_reference_and_partial_controls(source):
    assert not sc._qualified_parenthesized_legal_outline_markers(
        source, corpus_citation_path="us-ar/manual/example"
    )


@settings(max_examples=40, deadline=None)
@given(
    outer=st.sampled_from(["f", "D"]),
    root=st.integers(min_value=1, max_value=4),
    separator=st.sampled_from(["\n", "\n\n", "\r\n", "\r\n\r\n"]),
)
def test_attached_extension_siblings_preserve_partial_manual_source_ownership(
    outer, root, separator
):
    source = separator.join(
        [
            f"({outer}) Income calculations:",
            f"({root})(A) If the participant is paid weekly:",
            "(i) Total gross income.",
            "(ii)(a) Divide gross income by four.",
            "(b) Apply the deduction.",
            "(iii) Multiply the amount by 4.334.",
            "(B) Note. For five checks, divide by five.",
            f"({root + 1})(A) The next calculation.",
        ]
    )
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us-ar/manual/example"
    )
    paths = [branch.path for branch in branches]
    assert len(paths) == len(set(paths))
    parent = (outer.lower(), str(root), "a")
    by_path = {branch.path: branch for branch in branches}
    for path, literal in [
        ((*parent, "ii", "a"), "Divide gross income by four."),
        ((*parent, "ii", "b"), "Apply the deduction."),
        ((*parent, "iii"), "Multiply the amount by 4.334."),
        ((outer.lower(), str(root), "b"), "For five checks, divide by five."),
        ((outer.lower(), str(root + 1), "a"), "The next calculation."),
    ]:
        assert path in by_path
        branch = by_path[path]
        assert literal in sc._source_branch_direct_text(branch, branches=branches)
        assert source[branch.start : branch.end].strip() == branch.text


# Exact pinned Iowa bodies authenticate newly retained numeric-parent edges.
IA_NET_INCOME_SOURCE = """\
The term “net income” means the taxable income as properly computed for federal income tax purposes under section 63 of the Internal Revenue Code, with the following adjustments:
1. Subtract interest and dividends from federal securities.
2. Add interest and dividends from foreign securities and from securities of state and other political subdivisions exempt from federal income tax under the Internal Revenue Code, except for those securities the interest and dividends from which are exempt from taxation by the state of Iowa as otherwise provided by law, including:
a. Vision Iowa program bonds pursuant to section 12.71, subsection 8, Code 2020.
b. School infrastructure program bonds pursuant to section 12.81, subsection 8.
c. Iowa jobs program revenue bonds pursuant to section 12.87, subsection 8.
d. Iowa utilities commission and Iowa consumer advocate building project bonds pursuant to section 12.91, subsection 9.
e. Iowa finance authority beginning farmer loan program bonds pursuant to section 16.64, subsection 2.
f. Water pollution control works and drinking facilities financing program bonds pursuant to section 16.131, subsection 5.
g. Iowa prison infrastructure revenue bonds pursuant to section 12.80, subsection 3, and section 16.177, subsection 8.
h. Quad cities interstate metropolitan authority bonds pursuant to section 28A.24.
i. Iowa finance authority 911 program bonds pursuant to section 34A.20, subsection 6.
j. Soil and water conservation subdistrict bonds pursuant to section 161A.22.
k. Community college residence hall and dormitory bonds pursuant to section 260C.61.
l. Community college bond program bonds pursuant to section 260C.71, subsection 6.
m. Higher education loan authority bonds pursuant to section 261A.27.
n. State board of regents bonds pursuant to sections 262.41, 262.51, 262.60, 262A.8, and
263A.6.
o. Interstate bridges bonds pursuant to section 313A.36.
p. Aviation authority bonds pursuant to section 330A.16.
q. County health center bonds pursuant to section 331.441, subsection 2, paragraph “c”, subparagraph (7).
r. Rural water district bonds pursuant to section 357A.15.
s. Urban renewal bonds pursuant to section 403.9, subsection 2.
t. Municipal housing project bonds pursuant to section 403A.12.
u. Natural hazard mitigation financing program bonds pursuant to section 16.232, subsection 4.
3. Add any federal net operating loss deduction carried over from a taxable year beginning prior to January 1, 2023.
4. Individual taxpayers and married taxpayers who file a joint return or separate returns for Iowa income tax purposes may avail themselves of the disability income exclusion and shall compute the amount of the disability income exclusion subject to the limitations for joint federal income tax return filers provided by section 105(d) of the Internal Revenue Code. The disability income exclusion provided in section 105(d) of the Internal Revenue Code, as amended up to and including December 31, 1982, continues to apply for state income tax purposes for tax years beginning on or after January 1, 1984.
5. a. For tax years beginning in the 2023 calendar year, subtract the amount of federal income taxes paid during the tax year to the extent payment is for a tax year beginning prior to January 1, 2023, and add any federal income tax refunds received during the tax year to the extent the federal income tax was deducted for a tax year beginning prior to January 1,
2023. Federal income taxes paid for a tax year in which an Iowa return was not required to be filed shall not be subtracted.
b. Notwithstanding any other provision of law to the contrary, amounts subtracted or added pursuant to this subsection shall not be included in the calculation of net income for purposes of section 422.5, subsection 2 or 3, or section 422.13.
6. a. If the adjusted gross income includes income or loss from a small business operated by the taxpayer, an additional deduction shall be allowed in computing the income or loss from the small business if the small business hired for employment in the state during its annual accounting period ending with or during the taxpayer’s tax year any of the following:
(1) An individual with a disability domiciled in this state at the time of the hiring who meets any of the following conditions:
(a) Has a physical or mental impairment which substantially limits one or more major life activities.
(b) Has a record of that impairment.
(c) Is regarded as having that impairment.
(2) An individual domiciled in this state at the time of the hiring who meets any of the following conditions:
(a) Has been convicted of a felony in this or any other state or the District of Columbia.
(b) Is on parole pursuant to chapter 906.
(c) Is on probation pursuant to chapter 907, for an offense other than a simple misdemeanor.
(d) Is in a work release program pursuant to chapter 904, subchapter IX.
(3) An individual, whether or not domiciled in this state at the time of the hiring, who is on parole or probation and to whom the interstate compact for adult offender supervision under chapter 907B applies.
b. (1) The amount of the additional deduction is equal to sixty-five percent of the wages paid to individuals, but shall not exceed twenty thousand dollars per individual, named in paragraph “a”, subparagraphs (1), (2), and (3) who were hired for the first time by that business during the annual accounting period for work done in the state. This additional deduction is allowed for the wages paid to those individuals successfully completing a probationary period during the twelve months following the date of first employment by the business and shall be deducted at the close of the annual accounting period.
(2) The additional deduction shall not be allowed for wages paid to an individual who was hired to replace an individual whose employment was terminated within the twelve-month period preceding the date of first employment. However, if the individual being replaced left employment voluntarily without good cause attributable to the employer or if the individual was discharged for misconduct in connection with the individual’s employment as determined by the department of workforce development, the additional deduction shall be allowed.
(3) A taxpayer who is a partner of a partnership or a shareholder of a subchapter S corporation, may deduct that portion of wages qualified under this subsection paid by the partnership or subchapter S corporation based on the taxpayer’s pro rata share of the profits or losses from the partnership or subchapter S corporation.
c. For purposes of this subsection:
(1) “Physical or mental impairment” means any physiological disorder or condition, cosmetic disfigurement, or anatomical loss affecting one or more of the body systems or any mental or psychological disorder, including intellectual disability, organic brain syndrome, emotional or mental illness, and specific learning disabilities.
(2) (a) “Small business” means a profit or nonprofit business, including but not limited to an individual, partnership, corporation, joint venture, association, or cooperative, to which the following apply:
(i) It is not an affiliate or subsidiary of a business dominant in its field of operation.
(ii) It has twenty or fewer full-time equivalent positions and not more than the equivalent of three million dollars in annual gross revenues as computed for the preceding fiscal year or as the average of the three preceding fiscal years.
(iii) It does not include the practice of a profession.
(b) “Small business” includes an employee-owned business which has been an employee-owned business for less than three years or which meets the conditions of subparagraph division (a), subparagraph subdivisions (i) through (iii).
(c) For purposes of this definition, “dominant in its field of operation” means having more than twenty full-time equivalent positions and more than three million dollars in annual gross revenues, and “affiliate or subsidiary of a business dominant in its field of operation” means a business which is at least twenty percent owned by a business dominant in its field of operation, or by partners, officers, directors, majority stockholders, or their equivalents, of a business dominant in that field of operation.
7. a. If the adjusted gross income includes income or loss from a business operated by the taxpayer, and if the business does not qualify for the adjustment under subsection 6, an additional deduction shall be allowed in computing the income or loss from the business if the business hired for employment in the state during its annual accounting period ending with or during the taxpayer’s tax year either of the following:
(1) An individual domiciled in this state at the time of the hiring who meets any of the following conditions:
(a) Has been convicted of a felony in this or any other state or the District of Columbia.
(b) Is on parole pursuant to chapter 906.
(c) Is on probation pursuant to chapter 907, for an offense other than a simple misdemeanor.
(d) Is in a work release program pursuant to chapter 904, subchapter IX.
(2) An individual, whether or not domiciled in this state at the time of the hiring, who is on parole or probation and to whom the interstate compact for adult offender supervision under chapter 907B applies.
b. The amount of the additional deduction is equal to sixty-five percent of the wages paid to individuals, but shall not exceed twenty thousand dollars per individual, named in paragraph “a”, subparagraphs (1) and (2) who were hired for the first time by that business during the annual accounting period for work done in the state. This additional deduction is allowed for the wages paid to those individuals successfully completing a probationary period during the twelve months following the date of first employment by the business and shall be deducted at the close of the annual accounting period.
c. The additional deduction shall not be allowed for wages paid to an individual who was hired to replace an individual whose employment was terminated within the twelve-month period preceding the date of first employment. However, if the individual being replaced left employment voluntarily without good cause attributable to the employer or if the individual was discharged for misconduct in connection with the individual’s employment as determined by the department of workforce development, the additional deduction shall be allowed.
d. A taxpayer who is a partner of a partnership or a shareholder of a subchapter S corporation, may deduct that portion of wages qualified under this subsection paid by the partnership or subchapter S corporation based on the taxpayer’s pro rata share of the profits or losses from the partnership or subchapter S corporation.
e. The department shall develop and distribute information concerning the deduction available for businesses employing persons named in paragraph “a”, subparagraphs (1) and
(2).
8. Subtract, to the extent included, the amount of social security benefits taxable under section 86 of the Internal Revenue Code.
9. Add interest and dividends from regulated investment companies exempt from federal income tax under the Internal Revenue Code and subtract the loss on the sale or exchange of a share of a regulated investment company held for six months or less to the extent the loss was disallowed under section 852(b)(4)(B) of the Internal Revenue Code.
10. a. Subtract, to the extent included, the amount of a federal, state, or local grant provided to a communications service provider, if the grant is used to install broadband infrastructure that facilitates broadband service in targeted service areas at or above the download and upload speeds.
b. As used in this subsection, “broadband infrastructure”, “communications service provider”, and “targeted service area” mean the same as defined in section 8.76, respectively.
11. Add, to the extent deducted for federal tax purposes, charitable contributions under section 170 of the Internal Revenue Code to the extent such contribution was made to an organization for the purpose of deposit in the Iowa education savings plan trust established in chapter 12D, and the taxpayer designated that any part of the contribution be used for the direct benefit of any dependent of the taxpayer or any other single beneficiary designated by the taxpayer.
12. a. Subtract, to the extent included, income resulting from the payment by an employer of the taxpayer, whether paid to the taxpayer or to a lender, of principal or interest on any qualified education loan incurred by the taxpayer.
b. If the taxpayer has a deduction in computing federal taxable income under section 221 of the Internal Revenue Code for interest on a qualified education loan, the taxpayer shall recompute for purposes of this subsection the amount of the deduction under paragraph “a” by not subtracting any amount of income resulting from the employer’s payment of interest on a qualified education loan that was also deducted by the taxpayer under section 221 of the Internal Revenue Code.
c. For purposes of this subsection, “qualified education loan” means the same as defined in section 221 of the Internal Revenue Code.
13. a. For purposes of this subsection:
(1) “Farming business” means the production, care, growing, harvesting, preservation, handling, or storage of crops or forest or fruit trees; the production, care, feeding, management, and housing of livestock; or horticulture, all for intended profit.
(2) “Held” shall be determined with reference to the holding period provisions of section 1223 of the Internal Revenue Code and the federal regulations pursuant thereto.
(3) “Livestock” means the same as defined in section 717.1.
(4) “Materially participated” means the same as “material participation” in section 469(h) of the Internal Revenue Code, except that section 469(h)(3) of the Internal Revenue Code shall not apply.
(5) (a) “Real property used in a farming business” means all tracts of land and the improvements and structures located on such tracts which are in good faith used primarily for a farming business. Buildings which are primarily used or intended for human habitation are deemed to be used in a farming business when the building is located on or adjacent to the parcel used in the farming business. Land and the nonresidential improvements and structures located on such land that shall be considered to be used primarily in a farming business include but are not limited to land, improvements, or structures used for the storage or maintenance of farm machinery or equipment, for the drying, storage, handling, or preservation of agricultural crops, or for the storage of farm inputs, feed, or manure. Real property used in a farming business shall also include woodland, wasteland, pastureland, and idled land used for the conservation of natural resources including soil and water.
(b) Real property classified as agricultural property for Iowa property tax purposes, except real property described in section 441.21, subsection 12, paragraph “a” or “b”, shall be presumed to be real property used in a farming business. However, this presumption is rebuttable if the department shows by a preponderance of evidence that the real property did not meet the requirements of subparagraph division (a).
(6) “Relative” means a person that satisfies one or more of the following conditions:
(a) The individual is related to the taxpayer by consanguinity or affinity within the second degree as determined by common law.
(b) The individual is a lineal descendent of the taxpayer. For purposes of this subparagraph division, “lineal descendent” means children of the taxpayer, including legally adopted children and biological children, stepchildren, grandchildren, great-grandchildren, and any other lineal descendent of the taxpayer.
(c) An entity in which an individual who satisfies the conditions of either subparagraph division (a) or (b) has a legal or equitable interest as an owner, member, partner, or beneficiary.
(7) “Retired farmer” means an individual who is disabled or who is fifty-five years of age or older and who no longer materially participates in a farming business when an exclusion and deduction is claimed under this subsection.
b. Subtract the net capital gain from the sale of real property used in a farming business if one of the following conditions are satisfied:
(1) The taxpayer has materially participated in a farming business for a minimum of ten years and has held the real property used in a farming business for a minimum of ten years. If the taxpayer is a retired farmer, the taxpayer is considered to meet the material participation requirement if the taxpayer materially participated in a farming business for ten years or more in the aggregate, prior to making an election under this subsection.
(2) The taxpayer has held the real property used in a farming business which is sold to a relative of the taxpayer.
c. For a taxpayer who is a retired farmer, subtract the net capital gain from the sale of cattle or horses held by the taxpayer for breeding, draft, dairy, or sporting purposes for a period of twenty-four months or more from the date of acquisition; but only if the taxpayer materially participated in the farming business for five of the eight years preceding the farmer’s retirement or disability and who has sold all or substantially all of the taxpayer’s interest in the farming business by the time the election under this paragraph is made.
d. For a taxpayer who is a retired farmer, subtract the net capital gain from the sale of breeding livestock, other than cattle and horses, if the livestock is held by the taxpayer for a period of twelve months or more from the date of acquisition; but only if the taxpayer materially participated in the farming business for five of the eight years preceding the farmer’s retirement or disability and has sold all or substantially all of the taxpayer’s interest in the farming business by the time the election under this paragraph is made.
e. A taxpayer who is a retired farmer may make, subject to the limitations described in paragraphs “f” and “g”, a single, lifetime election to exclude all qualifying capital gains under paragraphs “b”, “c”, and “d”.
f. A taxpayer who is a retired farmer who elects to exclude capital gains under paragraph “b”, “c”, or “d” shall not claim the beginning farmer tax credit under section 422.11E or the exclusion for net income received pursuant to a farm tenancy agreement in subsection 14, in the tax year in which this election is made or in any subsequent year.
g. A taxpayer who is a retired farmer who claims the beginning farmer tax credit under section 422.11E shall not, in the same year, make an election under this subsection. A taxpayer who is a retired farmer and who elects to exclude the net income received from a farm tenancy agreement under subsection 14, shall not, in the same tax year or in any subsequent tax year, make the election under this subsection.
h. Married individuals who file separate state income tax returns shall allocate their combined annual net capital gain exclusion under paragraphs “b”, “c”, and “d” to each spouse in the proportion that each spouse’s respective net capital gain bears to the total net capital gain.
i. The department shall establish criteria, by rule, relating to whether and how a surviving spouse may claim the income exclusion for which a deceased retired farmer would have been eligible under this subsection.
14. a. Subtract, to the extent included, net income received by an eligible individual pursuant to a farm tenancy agreement covering real property held by the eligible individual for ten or more years, if the eligible individual materially participated in a farming business for ten or more years in the aggregate.
b. An individual who elects to exclude income received pursuant to a farm tenancy agreement under this subsection shall not claim any of the following in the tax year in which the election is made or in any succeeding year:
(1) The capital gain exclusion under subsection 13.
(2) The beginning farmer tax credit under section 422.11E.
c. Married individuals who file separate state income tax returns shall allocate their combined annual exclusion limit to each spouse in the proportion that each spouse’s respective net income from a farm tenancy agreement bears to the total net income from a farm tenancy agreement.
d. The department shall establish criteria, by rule, relating to whether and how a surviving spouse may claim the income exclusion for which a deceased eligible individual would have been eligible under this subsection.
e. Net income from a farm tenancy agreement earned, received, or reported by an entity taxed as a partnership for federal tax purposes, an S corporation, or a trust or estate is not eligible for the election and deduction in this subsection, even if such net income ultimately passes through to an eligible individual.
f. For purposes of this subsection:
(1) “Eligible individual” means an individual who is disabled or who is fifty-five years of age or older at the time the election is made, who no longer materially participates in a farming business at the time the election is made, and who, as an owner-lessor, is party to a farm tenancy agreement.
(2) “Farm tenancy agreement” means a written agreement outlining the rights and obligations of an owner-lessor and a tenant-lessee where the tenant-lessee has a farm tenancy as defined in section 562.1A. A “farm tenancy agreement” includes cash leases, crop share leases, or livestock share leases.
(3) “Farming business” means the production, care, growing, harvesting, preservation, handling, or storage of crops or forest or fruit trees; the production, care, feeding, management, and housing of livestock; or horticulture, all intended for profit.
(4) “Held” shall be determined with reference to the holding period provisions of section 1223 of the Internal Revenue Code and the federal regulations pursuant thereto.
(5) “Livestock” means the same as defined in section 717.1.
(6) “Materially participated” means the same as “material participation” in section 469(h) of the Internal Revenue Code, except that section 469(h)(3) of the Internal Revenue Code shall not apply.
15. Subtract, to the extent included, the amount of federal Segal AmeriCorps education award payments.
16. Subtract, to the extent included, payments received by an individual providing unskilled in-home health-related care services pursuant to section 249.3, subsection 2, paragraph “a”, subparagraph (2), to a member of the individual caregiver’s family. For purposes of this subsection, a member of the individual caregiver’s family includes a spouse, parent, stepparent, child, stepchild, brother, stepbrother, sister, stepsister, lineal ancestor, or lineal descendant, and such persons by marriage or adoption. A health care professional licensed by an examination board designated in section 147.13, subsections 1 through 10, is not eligible for the exemption authorized in this subsection.
17. If the taxpayer is owner of an individual development account certified under chapter 541A at any time during the tax year, deductions of all of the following shall be allowed:
a. Contributions made to the account by persons and entities, other than the taxpayer, as authorized in chapter 541A.
b. The amount of any state match payments authorized under section 541A.3, subsection
1.
c. Earnings from the account.
18. a. For a taxpayer who is sixty-five years of age or older and whose net income is less than one hundred thousand dollars, subtract, to the extent not otherwise deducted in computing federal taxable income, the amounts paid by the taxpayer for the purchase of health benefits coverage or insurance for the taxpayer or taxpayer’s spouse or dependent.
b. For purposes of this subsection, “net income” means net income as properly computed under this section without regard to the deduction in this subsection and with the following additional adjustments:
(1) Add back any amount of pensions or other retirement income received from any source which is not taxable under this subchapter, including but not limited to amounts deductible under subsections 8, 19, 20, and 21.
(2) Add back any amount of itemized or standard deduction, personal exemption deduction, or qualified business income deduction that was allowed as a deduction from federal adjusted gross income in computing federal taxable income under the Internal Revenue Code.
19. a. Subtract, to the extent included, the total amount received from a governmental or other pension or retirement plan, including defined benefit or defined contribution plans, annuities, individual retirement accounts, plans maintained or contributed to by an employer, or maintained or contributed to by a self-employed person as an employer, and deferred compensation plans or any earnings attributable to the deferred compensation plans received by a person who is any of the following:
(1) Disabled.
(2) Fifty-five years of age or older.
(3) The surviving spouse of an individual or a survivor having an insurable interest in an individual who would have qualified for the exemption under this subsection for the tax year.
b. Married taxpayers who file separate state income tax returns shall allocate their combined annual exclusion amount to each spouse in the proportion that each spouse’s respective income received from a pension or retirement plan bears to the total combined pension or retirement pay received.
c. (1) A taxpayer who is not disabled or fifty-five years of age or older and who receives a pension amount provided pursuant to section 97B.49B or 97B.49C, or chapter 97A, 410, or 411 as a surviving spouse may exclude the amount received from the pension in the tax year as a result of the death of the decedent, regardless of whether the decedent would have qualified for the exemption in the tax year.
(2) A taxpayer who is not disabled or fifty-five years of age or older and who receives other pension or retirement pay not specified in subparagraph (1) as a surviving spouse or as a survivor with an insurable interest in an individual who would have qualified for the exemption for the tax year may only exclude the amount received from a pension or retirement plan in the tax year as a result of the death of the decedent.
20. a. Subtract, to the extent included, retirement pay received by a taxpayer from the federal government for military service performed in the armed forces, the armed forces military reserve, or national guard.
b. The exclusion of retirement pay under this subsection is in addition to any exclusion provided under subsection 19.
21. a. Subtract, to the extent included, amounts received as survivor benefits by a taxpayer from the federal government pursuant to 10 U.S.C. §1447, et seq.
b. The exclusion of survivor benefits under this subsection is in addition to any exclusion provided under subsection 19.
22. a. Subtract the maximum contribution that may be deducted for Iowa income tax purposes as a participant in the Iowa educational savings plan trust pursuant to section 12D.3, subsection 1. For purposes of this paragraph, a participant who makes a contribution on or before the date prescribed in section 422.21 for making and filing an individual income tax return, excluding extensions, or the date for making and filing an individual income tax return determined by the director pursuant to an order issued under section 421.17, subsection 30, may elect to be deemed to have made the contribution on the last day of the preceding calendar year. The director, after consultation with the treasurer of state, shall prescribe by rule the manner and method by which a participant may make an election authorized by the preceding sentence.
b. Add the amount resulting from the cancellation of a participation agreement refunded to the taxpayer as a participant in the Iowa educational savings plan trust to the extent previously deducted as a contribution to the trust.
c. (1) Add, to the extent previously deducted as a contribution to the trust, the amount resulting from a withdrawal or transfer made by the taxpayer from the Iowa educational savings plan trust for purposes other than any of the following:
(a) The payment of qualified higher education expenses.
(b) The payment of tuition to an elementary or secondary school if the tuition amounts are qualified education expenses.
(c) A change in beneficiaries under, or transfer to another account within, the Iowa educational savings plan trust, or a transfer to the Iowa ABLE savings plan trust, provided such change or transfer is permitted under section 12D.6, subsection 5.
(d) The payment of expenses for fees, books, supplies, and equipment required for the participation of a beneficiary in an apprenticeship program.
(e) The payment of qualified education loan repayments.
(f) (i) A recontribution of a refund of any qualified higher education expenses from an eligible educational institution to the extent that such refund has been recontributed to the Iowa educational savings plan trust described in chapter 12D and meets all of the following criteria:
(A) The recontribution is made to the same account from which the original withdrawal was made.
(B) The recontribution occurs within sixty days of the date of refund.
(C) The recontribution amount does not exceed the amount refunded by the eligible educational institution.
(ii) A deduction under paragraph “a” shall not be taken for the amount of the recontribution.
(g) A direct trustee-to-trustee transfer to a Roth individual retirement account in accordance with the rules under section 529(c)(3)(E) of the Internal Revenue Code.
(2) For purposes of this paragraph:
(a) “Apprenticeship program” means a program registered and certified with the United States secretary of labor under section 1 of the National Apprenticeship Act, 29 U.S.C. §50.
(b) “Elementary or secondary school” means all of the following:
(i) An elementary or secondary school in this state which is accredited under section
256.11, and adheres to the provisions of the federal Civil Rights Act of 1964 and chapter 216.
(ii) An elementary or secondary school located out of state that educates a beneficiary who meets the definition of “children requiring special education” in section 256B.2, if the elementary or secondary school is accredited under the laws of the state in which it is located and adheres to the federal Civil Rights Act of 1964 and applicable state law analogous to chapter 216.
(c) “Qualified education expenses” and “tuition” all mean the same as defined in section
12D.1, subsection 2.
(d) “Qualified education loan” means the same as defined in section 12D.1, subsection 2.
(e) “Qualified education loan repayments” means amounts paid as principal or interest on any qualified education loan of the beneficiary or a sibling of the beneficiary. The repayment amounts shall not exceed ten thousand dollars in the aggregate for the beneficiary or the sibling, respectively.
(f) (i) “Qualified higher education expenses” means the same as defined in section 529(e)(3) of the Internal Revenue Code.
(ii) For purposes of this subparagraph division (c), “Internal Revenue Code” means the Internal Revenue Code of 1954, prior to the date of its redesignation as the Internal Revenue Code of 1986 by the Tax Reform Act of 1986, or means the Internal Revenue Code of 1986 as amended and in effect on January 1, 2020. This definition shall not be construed to include any amendment to the Internal Revenue Code enacted after the date specified in the preceding sentence, including any amendment with retroactive applicability or effectiveness.
(g) “Sibling” means the same as defined in section 12D.1, subsection 2.
23. Subtract, to the extent included, income from interest and earnings received from the Iowa educational savings plan trust created in chapter 12D.
24. a. (1) Subtract the amount contributed during the tax year on behalf of a designated beneficiary that is a resident of this state to the Iowa ABLE savings plan trust or to the qualified ABLE program with which the state has contracted pursuant to section 12I.10, not to exceed the maximum contribution level established in section 12I.3, subsection 1, paragraph “d”, or section 12I.10, subsection 2, paragraph “a”, as applicable.
(2) This paragraph “a” shall not apply to any amount of contribution that represents a transfer from the Iowa educational savings plan trust created in chapter 12D that meets the requirements of subsection 22, paragraph “c”, subparagraph (1), subparagraph division (c), and that was previously deducted as a contribution to the Iowa educational savings plan trust.
b. Add the amount resulting from the cancellation of a participation agreement refunded to the taxpayer as an account owner in the Iowa ABLE savings plan trust or the qualified ABLE program with which the state has contracted pursuant to section 12I.10 to the extent previously deducted pursuant to this subsection by the taxpayer or any other person as a contribution to the trust or qualified ABLE program, or to the extent the amount was previously deducted by the taxpayer or any other person pursuant to subsection 22, paragraph “a”, and qualified as a transfer under paragraph “a”, subparagraph (2), of this subsection.
c. Add the amount resulting from a withdrawal made by a taxpayer from the Iowa ABLE savings plan trust or the qualified ABLE program with which the state has contracted pursuant to section 12I.10 for purposes other than the payment of qualified disability expenses to the extent previously deducted pursuant to this subsection by the taxpayer or any other person as a contribution to the trust or qualified ABLE program, or to the extent the amount was previously deducted by the taxpayer or any other person pursuant to subsection 22, paragraph “a”, and qualified as a transfer under paragraph “a”, subparagraph
(2), of this subsection.
25. Subtract, to the extent included, income from interest and earnings received from the Iowa ABLE savings plan trust created in chapter 12I, or received by a resident account owner from a qualified ABLE program with which the state has contracted pursuant to section 12I.10.
26. Subtract, to the extent not otherwise excluded, the amount of withdrawals from qualified retirement plan accounts made during the tax year if the taxpayer or taxpayer’s spouse is a member of the Iowa national guard or reserve forces of the United States who is ordered to national guard duty or federal active duty. In addition, a penalty for such withdrawals shall not be assessed by the state.
27. a. Subject to the restrictions in paragraph “b”, subtract the sum of the following amounts:
(1) The amount of contributions made by an account holder during the tax year to the account holder’s first-time homebuyer savings accounts, not to exceed the following annual limit:
(a) (i) For married taxpayers who file a joint return and maintain a joint first-time homebuyer savings account, four thousand dollars.
(ii) For any other account holder, two thousand dollars.
(b) For the tax year beginning in the 2018 calendar year and for each subsequent tax year, the director shall multiply each dollar amount set forth in subparagraph division (a), subparagraph subdivisions (i) and (ii), by the latest cumulative inflation factor, shall round off the resulting product to the nearest one dollar, and shall incorporate the result into the income tax forms and instructions for each tax year. For purposes of this subparagraph division, “cumulative inflation factor” means the product of the annual inflation factor for the 2018 calendar year and all annual inflation factors for subsequent calendar years as determined by section 422.4, subsection 1, paragraph “a”. The cumulative inflation factor applies to all tax years beginning on or after January 1 of the calendar year for which the latest annual inflation factor has been determined. Notwithstanding any other provision, the annual inflation factor for the 2018 calendar year is one hundred percent.
(2) To the extent included, income from interest received from the account holder’s first-time homebuyer savings accounts.
b. (1) The subtraction in paragraph “a” shall not exceed the following aggregate lifetime limit:
(a) For married taxpayers who file a joint return and maintain a joint first-time homebuyer savings account, an amount equal to the product of the deductible amount determined for the year in paragraph “a”, subparagraph (1), subparagraph division (a), subparagraph subdivision (i), multiplied by ten.
(b) For any other account holder, an amount equal to the product of the deductible amount determined for the year in paragraph “a”, subparagraph (1), subparagraph division
(a), subparagraph subdivision (ii), multiplied by ten.
(2) The subtraction in paragraph “a” shall not be allowed to an account holder upon one of the following dates, whichever occurs first:
(a) January 1 of the tenth calendar year after the calendar year during which the account holder first opened a first-time homebuyer savings account.
(b) The date on which funds within an account holder’s first-time homebuyer savings account are withdrawn for purposes other than the payment or reimbursement of the designated beneficiary’s eligible home costs in connection with a qualified home purchase. Any amount transferred between different first-time homebuyer savings accounts of the same account holder by a person other than the account holder shall not be considered a withdrawal for purposes of this subparagraph division (b).
c. (1) Add, to the extent previously deducted under paragraph “a”, subparagraph (1), the amount withdrawn during the tax year from an account holder’s first-time homebuyer savings account for purposes other than the payment or reimbursement of the designated beneficiary’s eligible home costs in connection with a qualified home purchase.
(2) For purposes of this paragraph “c”, any amount remaining in an account holder’s first-time homebuyer savings account on January 1 of the tenth calendar year after the calendar year during which the account holder first opened a first-time homebuyer savings account shall be considered immediately withdrawn under subparagraph (1).
(3) For purposes of this paragraph “c”, the transfer of amounts between different first-time homebuyer accounts of the same account holder by a person other than the account holder shall not cause such transfer to be considered a withdrawal under subparagraph (1).
d. For any amount considered a withdrawal required to be added to net income pursuant to paragraph “c”, the account holder shall be assessed a penalty equal to ten percent of the amount of the withdrawal. The penalty shall not apply to withdrawals made by reason of the death of the account holder, or to withdrawals made pursuant to a garnishment, levy, or other order, including but not limited to an order in bankruptcy following a filing for protection under the federal bankruptcy code, 11 U.S.C. §101 et seq.
e. Add, to the extent deducted for federal tax purposes, interest, taxes, and other miscellaneous expenses to the extent such amounts are eligible home costs in connection with a qualified home purchase that were paid or reimbursed from funds in a first-time homebuyer savings account.
f. For purposes of this subsection, “account holder”, “designated beneficiary”, “eligible home costs”, “first-time homebuyer savings account”, and “qualified home purchase” mean the same as defined in section 541B.2.
28. Subtract, to the extent included, military student loan repayments received by the taxpayer serving on active duty in the national guard or armed forces military reserve or on active duty status in the armed forces.
29. Subtract, to the extent included, all pay received by the taxpayer from the federal government for military service performed while on active duty status in the armed forces, the armed forces military reserve, or the national guard.
30. a. If the taxpayer, while living, donates one or more of the taxpayer’s human organs to another human being for immediate human organ transplantation during the tax year, subtract, to the extent not otherwise excluded, the following unreimbursed expenses incurred by the taxpayer and related to the taxpayer’s organ donation:
(1) Travel expenses.
(2) Lodging expenses.
(3) Lost wages.
b. The maximum amount that may be deducted under paragraph “a” is ten thousand dollars. A taxpayer shall only take the deduction under this subsection once. If a deduction is taken under this subsection, the amount of expenses shall not be considered medical care expenses under section 213 of the Internal Revenue Code for state tax purposes.
c. For purposes of this subsection, “human organ” means all or part of a liver, pancreas, kidney, intestine, lung, or bone marrow.
31. Subtract, to the extent included, the amount of any grant provided pursuant to the injured veterans grant program pursuant to section 35A.14.
32. Subtract, to the extent included, amounts received from the veterans trust fund for any of the following items:
a. Travel expenses pursuant to section 35A.13, subsection 8, paragraph “a”.
b. Unemployment assistance pursuant to section 35A.13, subsection 8, paragraph “c”.
33. Subtract, to the extent not otherwise deducted in computing federal taxable income, the amounts paid by the taxpayer to the department of veterans affairs for the purpose of providing grants under the injured veterans grant program established in section 35A.14. Amounts subtracted under this subsection shall not be used by the taxpayer in computing the amount of charitable contributions as defined by section 170 of the Internal Revenue Code.
34. Subtract, to the extent included, income from interest and earnings received from a burial trust fund as defined in section 523A.102.
35. Subtract, to the extent included, the amount of victim compensation awards paid under the victim compensation program, victim restitution payments received pursuant to chapter 910 or 915, and any damages awarded by a court, and received by the taxpayer, in a civil action filed by the victim against the offender, during the tax year.
36. Subtract, to the extent included, the amount of any biodiesel production refund provided pursuant to section 423.4.
37. A taxpayer who is an eligible educator as defined in section 62(d)(1) of the Internal Revenue Code is allowed to take the deduction for certain expenses of elementary and secondary school teachers allowed under section 62(a)(2)(D) of the Internal Revenue Code in computing net income for state tax purposes in excess of the amount of the taxpayer’s deduction for certain expenses of elementary and secondary school teachers for federal tax purposes allowed under section 62(a)(2)(D) of the Internal Revenue Code, but not to exceed five hundred dollars.
38. Any adjustment subtracted from federal taxable income for an adjustment year pursuant to section 6225 of the Internal Revenue Code and the regulations thereunder shall be added back in computing net income of the partnership and the partners for state tax purposes for the adjustment year.
39. a. Section 163(j) of the Internal Revenue Code does not apply in computing net income for state tax purposes. If the taxpayer’s federal adjusted gross income for the tax year was increased or decreased by reason of the application of section 163(j) of the Internal Revenue Code, the taxpayer shall recompute net income for state tax purposes under rules prescribed by the director.
b. For any tax year in which paragraph “a” does not apply, a taxpayer shall not be permitted to deduct any amount of interest expense paid or accrued in a previous taxable year that is allowed as a deduction in the current taxable year by reason of the carryforward of disallowed business interest provisions of section 163(j)(2) of the Internal Revenue Code, if either of the following apply:
(1) The interest expense was originally paid or accrued during a tax year in which paragraph “a” applied.
(2) The interest expense was originally paid or accrued during a tax year in which the taxpayer was not required to file an Iowa return.
40. Notwithstanding any other provision of law to the contrary, any funds received by a student through a higher education institution to support the student’s financial needs as a result of the COVID-19 pandemic pursuant to §§3504, 18004, or 18008 of Pub. L. No. 116-136 shall not be included in the student’s Iowa net income for any tax year ending after March 27, 2020.
41. Reserved.
42. a. Subtract the following percentage of the net capital gain from the sale or exchange of capital stock of a qualified corporation for which an election is made by an employee-owner:
(1) For the tax year beginning in the 2023 calendar year, thirty-three percent.
(2) For the tax year beginning in the 2024 calendar year, sixty-six percent.
(3) For tax years beginning on or after January 1, 2025, one hundred percent.
b. (1) An employee-owner is entitled to make one irrevocable lifetime election to exclude the net capital gain from the sale or exchange of capital stock of one qualified corporation which capital stock was acquired by the employee-owner while employed and on account of employment by such qualified corporation.
(2) The election shall apply to all subsequent sales or exchanges of qualifying capital stock of the elected corporation within fifteen years of the date of the election, provided that the subsequent sales or exchanges were of capital stock in the same qualified corporation and were acquired by the employee-owner while employed and on account of employment by such qualified corporation.
(3) The election shall apply to qualifying capital stock that has been transferred by inter vivos gift from the employee-owner to the employee-owner’s spouse or to a trust for the benefit of the employee-owner’s spouse following the transfer. This subparagraph (3) shall apply to a spouse only if the spouse was married to the employee-owner on the date of the sale or exchange or the date of death of the employee-owner.
(4) If the employee-owner dies after having sold or exchanged qualifying capital stock without having made an election under this subsection, the surviving spouse or, if there is no surviving spouse, the personal representative of the employee-owner’s estate, may make the election that would have qualified under this subsection.
(5) The election shall be made in the manner and form prescribed by the department and shall be included with the taxpayer’s state income tax return for the taxable year in which the election is made.
c. For purposes of this subsection:
(1) “Capital stock” means common or preferred stock, either voting or nonvoting. “Capital stock” does not include stock rights, stock warrants, stock options, or debt securities.
(2) “Employee-owner” means an individual who owns capital stock in a qualified corporation for at least ten years, which capital stock was acquired by the individual while employed and on account of employment by such corporation for at least ten cumulative years.
(3) “Personal representative” means the same as defined in section 633.3, or if there is no such personal representative appointed, then the person legally authorized to perform substantially the same functions.
(4) (a) “Qualified corporation” means, with respect to an employee-owner, a corporation which, at the time of the first sale or exchange for which an election is made by the employee- owner under this subsection, meets all of the following conditions:
(i) The corporation employed individuals in this state for at least ten years.
(ii) The corporation has had at least five shareholders for the ten years prior to the first sale or exchange under this subsection.
(iii) The corporation has had at least two shareholders or groups of shareholders who are not related for the ten years prior to the first sale or exchange under this subsection. Two persons are considered related when, under section 318 of the Internal Revenue Code, one is a person who owns, directly or indirectly, capital stock that if directly owned would be attributed to the other person, or is the brother, sister, aunt, uncle, cousin, niece, or nephew of the other person who owns capital stock either directly or indirectly.
(b) “Qualified corporation” includes any member of an Iowa affiliated group if the Iowa affiliated group includes a member that has employed individuals in this state for at least ten years. For purposes of this subparagraph division, “Iowa affiliated group” means an affiliated group that has made a valid election to file an Iowa consolidated income tax return under section 422.37 in the year in which the deduction under this subsection is claimed. “Member” includes any entity included in the consolidated return under section 422.37, subsection 2, for the tax year in which the deduction is claimed.
(c) “Qualified corporation” also includes any corporation that was a party to a reorganization that was entirely or substantially tax free if such reorganization occurred during or after the employment of the employee-owner.
43. Subtract, to the extent included, the amount of an education savings account payment under section 257.11B received by the taxpayer for payment of qualified educational expenses.
44. a. For purposes of this subsection, “held” shall be determined with reference to the holding period provisions of section 1223 of the Internal Revenue Code and the federal regulations pursuant thereto.
b. Subtract the net capital gain from the following:
(1) The sale of cattle or horses held by the taxpayer for breeding, draft, dairy, or sporting purposes for a period of twenty-four months or more from the date of acquisition, but only if the taxpayer received more than one-half of the taxpayer’s gross income from farming or ranching operations during the tax year.
(2) The sale of breeding livestock, other than cattle or horses, held by the taxpayer for a period of twelve months or more from the date of acquisition, but only if the taxpayer received more than one-half of the taxpayer’s gross income from farming or ranching operations during the tax year.
c. A retired farmer as defined in subsection 13 is not eligible for the exclusion in this subsection if taking the exclusion allowed in subsection 13, paragraph “c” or “d”, in the same tax year.
45. a. Subtract, to the extent included, the amount of any qualifying state disaster recovery new housing grant issued to an individual or business by the economic development authority.
b. For purposes of this subsection, “qualifying state disaster recovery new housing grant” means an award of a state disaster recovery new housing grant that was applied for between August 20, 2024, and December 31, 2024, and approved and issued by the economic development authority.
c. This subsection is repealed January 1, 2028."""

IA_CORPORATE_TAX_SOURCE = """\
1. a. A tax is imposed annually upon each corporation doing business in this state, or deriving income from sources within this state, in an amount computed by applying the following rates of taxation to the net income received by the corporation during the income year:
(1) On the first twenty-five thousand dollars of taxable income, or any part thereof, the rate of six percent for tax years beginning prior to January 1, 2021, and the rate of five and one-half percent for tax years beginning on or after January 1, 2021.
(2) On taxable income between twenty-five thousand dollars and one hundred thousand dollars or any part thereof, the rate of eight percent for tax years beginning prior to January 1, 2021, and the rate of five and one-half percent for tax years beginning on or after January 1, 2021.
(3) On taxable income between one hundred thousand dollars and two hundred fifty thousand dollars or any part thereof, the rate of ten percent for tax years beginning prior to January 1, 2021, and the rate of nine percent for tax years beginning on or after January 1, 2021.
(4) On taxable income of two hundred fifty thousand dollars or more, the rate of twelve percent for tax years beginning prior to January 1, 2021, and the rate of nine and eight-tenths percent for tax years beginning on or after January 1, 2021.
b. (1) (a) Notwithstanding paragraph “a”, the department of management and the department of revenue shall determine corporate income tax rates as provided in this paragraph. A tax rate in this subsection shall remain in effect until the tax rate is adjusted pursuant to this paragraph.
(b) By November 1, 2022, and by November 1 each year thereafter, the department of management shall determine the net corporate income tax receipts for the fiscal year preceding the determination date. If net corporate income tax receipts for the preceding fiscal year exceed seven hundred million dollars, the department of revenue shall adjust and apply new corporate income tax rates as provided in subparagraph (2).
(2) (a) If a determination has been made that net corporate income tax receipts for the preceding fiscal year exceeded seven hundred million dollars, the department of revenue shall adjust the tax rates specified in paragraph “a”, subparagraphs (3) and (4), and apply the adjusted rates for tax years beginning on or after the next January 1 following the determination date.
(b) (i) The tax rates subject to adjustment shall be adjusted in such a way that when combined with all the other rates specified in paragraph “a”, the tax rates would have generated net corporate income tax receipts that equal seven hundred million dollars in the preceding fiscal year.
(ii) When adjusting the tax rates, the tax rates shall be adjusted as follows:
(A) The tax rate in effect that corresponds with the specified tax rate in paragraph “a”, subparagraph (4), shall first be adjusted but not below the tax rate in effect that corresponds with the specified rate in paragraph “a”, subparagraph (3).
(B) If after the adjustment in subparagraph part (A) is made, and an additional adjustment is necessary, the tax rates that correspond with the rates specified in paragraph “a”, subparagraphs (3) and (4), shall be adjusted on an equal basis.
(iii) The tax rates adjusted pursuant to this paragraph shall not be adjusted below five and one-half percent.
(iv) The tax rates, when adjusted, shall be rounded down to the nearest one-tenth of one percent.
(3) If a tax rate is adjusted pursuant to this paragraph, the director of revenue shall cause an advisory notice containing the new corporate tax rates to be published in the Iowa administrative bulletin and on the internet site of the department of revenue. The calculation and publication of the adjusted tax rate by the director of revenue is exempt from chapter 17A, and shall be submitted for publication by the first December 31 following the determination date to adjust the tax rates.
1A. There is imposed upon each corporation exempt from the general business tax on corporations by section 422.34, subsection 2, a tax at the rates in subsection 1 upon the state’s apportioned share computed in accordance with subsections 2 and 3 of the unrelated business income computed in accordance with the Internal Revenue Code and with the adjustments set forth in section 422.35.
2. a. If the trade or business of the corporation is carried on entirely within the state, the tax shall be imposed on the entire net income, but if the trade or business is carried on partly within and partly without the state or if income is derived from sources partly within and partly without the state, or if income is derived from trade or business and sources, all of which are not entirely in the state, the tax shall be imposed only on the portion of the net income reasonably attributable to the trade or business or sources within the state, with the net income attributable to the state to be determined as follows:
(1) Nonbusiness interest, dividends, rents and royalties, less related expenses, shall be allocated within and without the state in the following manner:
(a) Nonbusiness interest, dividends, and royalties from patents and copyrights shall be allocable to this state if the taxpayer’s commercial domicile is in this state.
(b) Nonbusiness rents and royalties received from real property located in this state are allocable to this state.
(c) Nonbusiness rents and royalties received from tangible personal property are allocable to this state to the extent that the property is utilized in this state; or in their entirety if the taxpayer’s commercial domicile is in this state and the taxpayer is not taxable in the state in which the property is utilized. The extent of utilization of tangible personal property in a state is determined by multiplying the rents and royalties by a fraction, the numerator of which is the number of days of physical location of the property in the state during the rental or royalty period in the taxable year and the denominator of which is the number of days of physical location of the property everywhere during all rental or royalty periods in the taxable year. If the physical location of the property during the rental or royalty period is unknown or unascertainable by the taxpayer, tangible personal property is utilized in the state in which the property was located at the time the rental or royalty payor obtained possession.
(d) Nonbusiness capital gains and losses from the sale or other disposition of assets shall be allocated as follows:
(i) Gains and losses from the sale or other disposition of real property located in this state are allocable to this state.
(ii) Gains and losses from the sale or other disposition of tangible personal property are allocable to this state if the property had a situs in this state at the time of the sale or disposition or if the taxpayer’s commercial domicile is in this state and the taxpayer is not taxable in the state in which the property had a situs.
(iii) Gains and losses from the sale or disposition of intangible personal property are allocable to this state if the taxpayer’s commercial domicile is in this state.
(2) Net nonbusiness income of the above class having been separately allocated and deducted as above provided, the remaining net business income of the taxpayer shall be allocated and apportioned as follows:
(a) Business interest, dividends, rents, and royalties shall be reasonably apportioned within and without the state under rules adopted by the director.
(b) Capital gains and losses from the sale or other disposition of assets shall be apportioned to the state based upon the business activity ratio applicable to the year the gain or loss is determined if the corporation determines Iowa taxable income by a sales, gross receipts or other business activity ratio. If the corporation has only allocable income, capital gains and losses from the sale or other disposition of assets shall be allocated in accordance with subparagraph (1), subparagraph division (d).
(c) Where income is derived from business other than the manufacture or sale of tangible personal property, the income shall be specifically allocated or equitably apportioned within and without the state under rules of the director.
(d) Where income is derived from the manufacture or sale of tangible personal property, the part attributable to business within the state shall be in that proportion which the gross sales made within the state bear to the total gross sales.
(e) (i) Notwithstanding subparagraph division (c), where income is derived by a broadcaster from broadcasting, the part attributable to business within the state shall be in the proportion that the gross receipts from broadcasting derived from customers whose commercial domicile is in this state bears to the total gross receipts from broadcasting.
(ii) Notwithstanding subparagraph subdivision (i) or subparagraph division (c), where income is derived by a broadcaster from national or local political advertising that is directed exclusively at one or more markets in this state, all gross receipts from such advertising shall be attributable to business within the state.
(iii) For purposes of this subparagraph division:
(A) “Broadcaster” means a taxpayer who is engaged in the business of broadcasting. “Broadcaster” includes a television network, a cable program network, and a television distribution company. “Broadcaster” does not include a cable system operator, a direct broadcast satellite system operator, or a television or radio station licensed by the federal communications commission.
(B) “Broadcasting” means the transmission of film programming by an electronic or other signal conducted by microwaves, wires, lines, coaxial cables, wave guides, fiber optics, satellite transmissions, or through any other means of communication directly or indirectly to viewers and listeners.
(C) “Customer” means a person who has a direct contractual relationship with a broadcaster from whom the broadcaster derives gross receipts. “Customer” includes but is not limited to an advertiser or licensee.
(D) “Gross receipts from broadcasting” means gross receipts of a broadcaster from transactions and activities in the regular course of its business, including but not limited to advertising, licensing, and distribution, but excluding gross receipts from the sale of real property or tangible personal property.
(f) Notwithstanding subparagraph division (c), income described in section 29C.24, subsection 3, paragraph “a”, subparagraph (3), shall not be allocated and apportioned to the state, as provided in section 29C.24.
(g) Where income consists of more than one class of income as provided in subparagraph divisions (a) through (e) of this subparagraph, it shall be reasonably apportioned by the business activity ratio provided in rules adopted by the director.
(h) The gross sales of the corporation within the state shall be taken to be the gross sales from goods delivered or shipped to a purchaser within the state regardless of the F.O.B. point or other conditions of the sale, excluding deliveries for transportation out of the state.
b. For the purpose of this subsection:
(1) “Manufacture” shall include the extraction and recovery of natural resources and all processes of fabricating and curing.
(2) “Sale” shall include exchange.
(3) “Tangible personal property” shall be taken to mean corporeal personal property, such as machinery, tools, implements, goods, wares, and merchandise, and shall not be taken to mean money deposits in banks, shares of stock, bonds, notes, credits, or evidence of an interest in property and evidences of debt.
3. If any taxpayer believes that the method of allocation and apportionment prescribed in subsections 1A and 2, as administered by the director and applied to the taxpayer’s business, has operated or will so operate as to subject the taxpayer to taxation on a greater portion of the taxpayer’s net income than is reasonably attributable to business or sources within the state, the taxpayer shall be entitled to file with the director a statement of the taxpayer’s objections and of such alternative method of allocation and apportionment as the taxpayer believes to be proper under the circumstances with such detail and proof and within such time as the director may reasonably prescribe; and if the director shall conclude that the method of allocation and apportionment theretofore employed is in fact inapplicable and inequitable, the director shall redetermine the taxable income by such other method of allocation and apportionment as seems best calculated to assign to the state for taxation the portion of the income reasonably attributable to business and sources within the state, not exceeding, however, the amount which would be arrived at by application of the statutory rules for apportionment.
4. Reserved.
5. a. The taxes imposed under this subchapter shall be reduced by a state tax credit through the tax year beginning on or after January 1, 2025, but before January 1, 2026, for increasing research activities in this state equal to the sum of the following:
(1) Six and one-half percent of the excess of qualified research expenses during the tax year over the base amount for the tax year based upon the state’s apportioned share of the qualifying expenditures for increasing research activities.
(2) Six and one-half percent of the basic research payments determined under section 41(e)(1)(A) of the Internal Revenue Code during the tax year based upon the state’s apportioned share of the qualifying expenditures for increasing research activities.
b. (1) The state’s apportioned share of the qualifying expenditures for increasing research activities is a percent equal to the ratio of qualified research expenditures in this state to the total qualified research expenditures.
(2) For the purpose of calculating the state’s apportioned share of the qualifying expenditures for increasing research activities in subparagraph (1), the following criteria shall apply only to the determination of qualified research expenditures in this state:
(a) Wages paid to an employee for qualified services, or contract research expenses paid to a third party for the performance of qualified research services, shall only constitute qualified research expenses in this state if the services are performed in this state, and if the following conditions are met, as applicable:
(i) For qualified services performed by employees, during the period of the tax year that the business is engaging in one or more research projects, a majority of the total services performed by the employee for the business are directly related to those research projects.
(ii) For the performance of qualified research services by a third party, during the period of the business’s tax year that the third party is performing research services for the business, a majority of the total services performed by the person for the third party are directly related to those research projects of the business.
(b) The substantially all rule for determining qualified services as described in section 41(b)(2)(B) of the Internal Revenue Code and Treas. Reg. 1.41-2(d)(2) does not apply.
(c) Amounts paid for the right to use computers as described in section 41(b)(2)(A)(iii) of the Internal Revenue Code shall not be qualified research expenses in this state.
(d) For tax years beginning on or after January 1, 2023, but before January 1, 2027, amounts paid for supplies as defined in section 41(b)(2)(C) of the Internal Revenue Code shall only constitute qualified research expenses in this state if the supplies directly relate to research performed in this state and shall be limited to the following allowable percentages:
(i) For the tax year beginning on or after January 1, 2023, but before January 1, 2024, eighty percent of the amounts paid for supplies directly related to research performed in this state.
(ii) For the tax year beginning on or after January 1, 2024, but before January 1, 2025, sixty percent of the amounts paid for supplies directly related to research performed in this state.
(iii) For the tax year beginning on or after January 1, 2025, but before January 1, 2026, forty percent of the amounts paid for supplies directly related to research performed in this state.
(e) For tax years beginning on or after January 1, 2026, amounts paid for supplies as defined in section 41(b)(2)(C) of the Internal Revenue Code shall not be qualified research expenses in this state.
c. In lieu of the credit amount computed in paragraph “a”, subparagraph (1), a corporation shall elect to compute the credit amount for qualified research expenses incurred in this state in a manner consistent with the alternative simplified credit described in section 41(c)(4) of the Internal Revenue Code if the taxpayer elected or was required to use the alternative simplified credit method for federal income tax purposes for the same taxable year.
d. For purposes of the alternate credit computation method in paragraph “c”, the following criteria shall apply:
(1) The credit percentages applicable to qualified research expenses described in section 41(c)(4)(A) and clause (ii) of section 41(c)(4)(B) of the Internal Revenue Code are four and fifty-five hundredths percent and one and ninety-five hundredths percent, respectively.
(2) Basic research payments and qualified research expenses shall only include amounts for research conducted in this state. A taxpayer’s qualified research expenses in this state and average prior year qualified research expenses in this state shall be determined in accordance with the rules in paragraph “b”, subparagraph (2).
e. A corporation shall only be eligible for the credit provided in this subsection if the business conducting the research meets all of the following requirements:
(1) (a) The business is engaged in the manufacturing, life sciences, agriscience, software engineering, or aviation and aerospace industry.
(b) Persons that shall not be considered to be engaged in the manufacturing, life sciences, agriscience, software engineering, or aviation and aerospace industry, and thus are not eligible for the credit, include but are not limited to all of the following:
(i) (A) A person engaged in agricultural production as defined in section 423.1, except if the credit is based on conducting agriscience research and the person or the business is engaged in bovine and porcine veterinary research, the person shall not be considered to be engaged in agricultural production as defined in section 423.1.
(B) As used in this subparagraph subdivision, “agriscience research” means research that is approved and overseen or monitored by a board that includes, at a minimum, an individual who was employed with, contracted by, or professionally trained by an accredited university as a researcher in an applied animal science and an individual holding a doctor of veterinary medicine or a doctoral degree in an applied animal science; is conducted in this state in an applied animal science; improves the scientific knowledge base or increases scientific innovation, performance, or viability within this state; the results of the research are evaluated by a person educated and trained in statistics by an accredited university and capable of applying generally accepted methodologies to the results in accordance with industry standards in an applied animal science; and the results of the research are made available to the public by submission to or publication in a journal, magazine, or similar periodical, if the statistical evaluation indicated the research is reliable and relevant to an applied animal science.
(C) As used in this subparagraph subdivision, “applied animal science” includes the areas of animal science, veterinary medicine, nutritional science, genetic science, and microbiology.
(ii) A person who is a contractor, subcontractor, builder, or a contractor-retailer that engages in commercial and residential repair and installation, including but not limited to heating or cooling installation and repair, plumbing and pipe fitting, security system installation, and electrical installation and repair. For purposes of this subparagraph subdivision, “contractor-retailer” means a business that makes frequent retail sales to the public or to other contractors and that also engages in the performance of construction contracts.
(iii) A finance or investment company.
(iv) A retailer.
(v) A wholesaler.
(vi) A transportation company.
(vii) A publisher.
(viii) An agricultural cooperative association as defined in section 502.102.
(ix) A real estate company.
(x) A collection agency.
(xi) An accountant.
(xii) An architect.
(2) The business claims and is allowed a research credit for such qualified research expenses under section 41 of the Internal Revenue Code for the same taxable year as it is claiming the credit provided in this subsection.
(3) The credit provided in this subsection is claimed on a return filed by the due date for filing the return, including extensions of time. If timely claimed, the business shall not increase the credit claim on an amended return or otherwise unless either of the following apply:
(a) The amended return is filed within six months of the due date for filing the return which includes extensions of time.
(b) The increase results from an audit or examination by the internal revenue service or the department.
f. (1) For purposes of this subsection, “base amount” means the product of the fixed-based percentage times the average annual gross receipts of the taxpayer for the four taxable years preceding the taxable year for which the credit is being determined, but in no event shall the base amount be less than fifty percent of the qualified research expenses for the credit year.
(2) For purposes of this subsection, “basic research payment” and “qualified research expense” mean the same as defined for the federal credit for increasing research activities under section 41 of the Internal Revenue Code, except as otherwise described in paragraph “b”, subparagraph (2), and paragraph “d”, subparagraph (2).
g. (1) (a) The following percentage of the credit in excess of the tax liability for the taxable year shall be refunded with interest in accordance with section 421.60, subsection 2, paragraph “e”:
(i) For the tax year beginning on or after January 1, 2023, but before January 1, 2024, ninety percent.
(ii) For the tax year beginning on or after January 1, 2024, but before January 1, 2025, eighty percent.
(iii) For the tax year beginning on or after January 1, 2025, but before January 1, 2026, seventy percent.
(iv) For the tax year beginning on or after January 1, 2026, but before January 1, 2027, sixty percent.
(b) In lieu of claiming a refund pursuant to this subparagraph, a taxpayer may elect to have the overpayment otherwise eligible for a refund shown on its final, completed return credited to the tax liability for the following taxable year.
(2) Commencing with tax years beginning on or after January 1, 2027, fifty percent of any credit in excess of the tax liability for the taxable year shall be refunded with interest in accordance with section 421.60, subsection 2, paragraph “e”. In lieu of claiming a refund, a taxpayer may elect to have the overpayment otherwise eligible for a refund shown on its final, completed return credited to the tax liability for the following taxable year.
(3) In applying the credit in this subsection against tax liability and computing the eligible refund amount, the credit shall be applied after all nonrefundable credits available to the taxpayer are applied, but before any other refundable credit available to the taxpayer is applied.
h. The department shall by February 15 of each year issue an annual report to the general assembly containing the total amount of all claims made by employers under this subsection and the portion of the claims issued as refunds, for all claims processed during the previous calendar year. The report shall contain the name of each claimant for whom a tax credit in excess of five hundred thousand dollars was issued and the amount of the credit received.
i. This subsection is repealed January 1, 2027.
6. a. The taxes imposed under this subchapter shall be reduced by a new jobs tax credit. An industry which has entered into an agreement under chapter 260E and which has increased its base employment level by at least ten percent within the time set in the agreement or, in the case of an industry without a base employment level, adds new jobs within the time set in the agreement is entitled to this new jobs tax credit for the tax year selected by the industry. In determining if the industry has increased its base employment level by ten percent or added new jobs, only those new jobs directly resulting from the project covered by the agreement and those directly related to those new jobs shall be counted.
b. The amount of this credit is equal to the product of six percent of the taxable wages, as defined in section 96.1A, subsection 36, upon which an employer is required to contribute to the state unemployment compensation fund, times the number of new jobs existing in the tax year that directly result from the project covered by the agreement or new jobs that directly result from those new jobs. The tax year chosen by the industry shall either begin or end during the period beginning with the date of the agreement and ending with the date by which the project is to be completed under the agreement. Any credit in excess of the tax liability for the tax year may be credited to the tax liability for the following ten tax years or until depleted in less than the ten years.
c. For purposes of this section, “agreement”, “industry”, “new job”, and “project” mean the same as defined in section 260E.2 and “base employment level” means the number of full-time jobs an industry employs at the plant site which is covered by an agreement under chapter 260E on the date of that agreement.
7. Reserved.
8. The taxes imposed under this subchapter shall be reduced by a franchise tax credit. A taxpayer who is a shareholder in a financial institution, as defined in section 581 of the Internal Revenue Code, which has in effect for the tax year an election under subchapter S of the Internal Revenue Code shall compute the amount of the tax credit by recomputing the amount of tax under this subchapter by reducing the taxable income of the taxpayer by the taxpayer’s pro rata share of the items of income and expense of the financial institution. This recomputed tax shall be subtracted from the tax computed under this subchapter and the resulting amount, which shall not exceed the taxpayer’s pro rata share of franchise tax paid by the financial institution, is the amount of the franchise tax credit allowed.
9. a. (1) The taxes imposed under this subchapter shall be reduced by an assistive device tax credit through the tax year beginning on or after January 1, 2024, but before January 1, 2025. A small business purchasing, renting, or modifying an assistive device or making workplace modifications for an individual with a disability who is employed or will be employed by the small business is eligible, subject to availability of credits, to receive this assistive device tax credit which is equal to fifty percent of the first five thousand dollars paid during the tax year for the purchase, rental, or modification of the assistive device or for making the workplace modifications. The following percentage of any credit in excess of the tax liability shall be refunded with interest in accordance with section 421.60, subsection 2, paragraph “e”, as follows:
(a) For the tax year beginning on or after January 1, 2023, but before January 1, 2024, ninety-five percent.
(b) For the tax year beginning on or after January 1, 2024, but before January 1, 2025, ninety percent.
(2) In lieu of claiming a refund, a taxpayer may elect to have the overpayment otherwise eligible for a refund shown on the taxpayer’s final, completed return credited to the tax liability for the following tax year. If the small business elects to take the assistive device tax credit, the small business shall not deduct for Iowa tax purposes any amount of the cost of an assistive device or workplace modifications which is deductible for federal income tax purposes.
b. To receive the assistive device tax credit, the eligible small business must submit an application to the economic development authority. If the taxpayer meets the criteria for eligibility, the economic development authority shall issue to the taxpayer a certification of entitlement for the assistive device tax credit. However, the combined amount of tax credits that may be approved for a fiscal year under this subsection shall not exceed five hundred thousand dollars. Tax credit certificates shall be issued on an earliest filed basis. The certification shall contain the taxpayer’s name, address, tax identification number, the amount of the credit, and tax year for which the certificate applies. The taxpayer must file the tax credit certificate with the taxpayer’s corporate income tax return in order to claim the tax credit. The economic development authority and department of revenue shall each adopt rules to jointly administer this subsection and shall provide by rule for the method to be used to determine for which fiscal year the tax credits are approved.
c. For purposes of this subsection:
(1) “Assistive device” means any item, piece of equipment, or product system which is used to increase, maintain, or improve the functional capabilities of an individual with a disability in the workplace or on the job. “Assistive device” does not mean any medical device, surgical device, or organ implanted or transplanted into or attached directly to an individual. “Assistive device” does not include any device for which a certificate of title is issued by the state department of transportation, but does include any item, piece of equipment, or product system otherwise meeting the definition of “assistive device” that is incorporated, attached, or included as a modification in or to such a device issued a certificate of title.
(2) “Disability” means the same as defined in section 15.102, except that it does not include alcoholism.
(3) “Small business” means a business that either had gross receipts for its preceding tax year of three million dollars or less or employed not more than fourteen full-time employees during its preceding tax year.
(4) “Workplace modifications” means physical alterations to the work environment.
d. This subsection is repealed January 1, 2031.
10. The taxes imposed under this subchapter shall be reduced by a historic preservation tax credit allowed under chapter 404A.
11. Reserved.
11A. Reserved.
11B. The taxes imposed under this subchapter shall be reduced by an E-85 gasoline promotion tax credit for each tax year that the taxpayer is eligible to claim the tax credit under this subsection.
a. The taxpayer shall claim the tax credit in the same manner as provided in section
422.11O. The taxpayer may claim the tax credit according to the same requirements, for the same amount, and calculated in the same manner, as provided for the E-85 gasoline promotion tax credit pursuant to section 422.11O.
b. Any E-85 gasoline promotion tax credit which is in excess of the taxpayer’s tax liability shall be refunded or may be shown on the taxpayer’s final, completed return credited to the tax liability for the following tax year in the same manner as provided in section 422.11O.
c. This subsection is repealed January 1, 2028.
11C. The taxes imposed under this subchapter shall be reduced by a biodiesel blended fuel tax credit for each tax year that the taxpayer is eligible to claim the tax credit under this subsection.
a. The taxpayer may claim the biodiesel blended fuel tax credit according to the same requirements, for the same amount, and calculated in the same manner, as provided for the biodiesel blended fuel tax credit pursuant to section 422.11P.
b. Any biodiesel blended fuel tax credit which is in excess of the taxpayer’s tax liability shall be refunded or may be shown on the taxpayer’s final, completed return credited to the tax liability for the following tax year in the same manner as provided in section 422.11P.
c. This subsection is repealed January 1, 2028.
11D. The taxes imposed under this subchapter shall be reduced by an E-15 plus gasoline promotion tax credit for each tax year that the taxpayer is eligible to claim the tax credit under this subsection.
a. The taxpayer shall claim the tax credit in the same manner as provided in section
422.11Y. The taxpayer may claim the tax credit according to the same requirements, for the same amount, and calculated in the same manner, as provided for the E-15 plus gasoline promotion tax credit pursuant to section 422.11Y.
b. Any E-15 plus gasoline promotion tax credit which is in excess of the taxpayer’s tax liability shall be refunded or may be shown on the taxpayer’s final, completed return credited to the tax liability for the following tax year in the same manner as provided in section 422.11Y.
c. This subsection is repealed January 1, 2028.
12. a. The taxes imposed under this subchapter shall be reduced by an investment tax credit authorized pursuant to section 15E.27 for an investment in a qualifying business.
b. The taxes imposed under this subchapter shall be reduced by investment tax credits authorized pursuant to sections 15.508 and 15.496.
13. The taxes imposed under this subchapter shall be reduced by an innovation fund investment tax credit allowed under section 15E.52.
14. The taxes imposed under this subchapter shall be reduced by an endow Iowa tax credit authorized pursuant to section 15E.305.
15. The taxes imposed under this subchapter shall be reduced by a workforce housing investment tax credit allowed under section 15.355, subsection 3.
16. The taxes imposed under this subchapter shall be reduced by tax credits for wind energy production allowed under chapter 476B and for renewable energy allowed under chapter 476C.
17. The taxes imposed under this subchapter shall be reduced by the research and development tax credit allowed pursuant to section 15.524.
18. Reserved.
19. Reserved.
20. The taxes imposed under this subchapter shall be reduced by a tax credit authorized pursuant to section 15E.66, if redeemed, for investments in the Iowa fund of funds.
21. The taxes imposed under this subchapter shall be reduced by a beginning farmer tax credit as allowed under chapter 16, subchapter VIII, part 5, subpart B.
22. The taxes imposed under this subchapter shall be reduced by a renewable chemical production tax credit allowed under section 15.319. This subsection is repealed January 1,
2041.
23. The taxes imposed under this subchapter shall be reduced by a sustainable aviation fuel tax credit allowed under section 15.533. This subsection is repealed January 1, 2037.
24. Reserved.
25. a. The taxes imposed under this subchapter shall be reduced by a charitable conservation contribution tax credit equal to fifty percent of the fair market value of a qualified real property interest located in the state that is conveyed as an unconditional charitable donation in perpetuity by the taxpayer to a qualified organization exclusively for conservation purposes. The maximum amount of tax credit is one hundred thousand dollars. The amount of the contribution for which the tax credit is claimed shall not be deductible in determining taxable income for state tax purposes.
b. For purposes of this section, “conservation purpose”, “qualified organization”, and “qualified real property interest” mean the same as defined for the qualified conservation contribution under section 170(h) of the Internal Revenue Code, except that a conveyance of land for open space for the purpose of fulfilling density requirements to obtain subdivision or building permits shall not be considered a conveyance for a conservation purpose.
c. Any credit in excess of the tax liability is not refundable but the excess for the tax year may be credited to the tax liability for the following twenty tax years or until depleted, whichever is the earlier.
26. The taxes imposed under this subchapter shall be reduced by a redevelopment tax credit allowed under chapter 15, subchapter II, part 9.
27. Reserved.
28. The taxes imposed under this subchapter shall be reduced by a school tuition organization tax credit allowed under section 422.11S.
29. a. The taxes imposed under this subchapter shall be reduced by a solar energy system tax credit equal to sixty percent of the federal energy credit related to solar energy systems provided in section 48(a)(2)(A)(i)(II) and section 48(a)(2)(A)(i)(III) of the Internal Revenue Code, not to exceed twenty thousand dollars. For installations occurring on or after January 1, 2016, the applicable percentage of the federal energy credit related to solar energy systems shall be fifty percent.
b. The taxpayer may claim the credit pursuant to this subsection according to the same requirements, conditions, and limitations as provided pursuant to section 422.11L.
30. The taxes imposed under this subchapter shall be reduced by a from farm to food donation tax credit as allowed under chapter 190B.
31. The taxes imposed under this subchapter shall be reduced by a Hoover presidential library tax credit allowed under section 15E.364.
32. a. The taxes imposed under this subchapter shall be reduced by an employer child care tax credit allowed pursuant to section 237A.31.
b. This subsection is repealed January 1, 2031."""

DEEP_DEFINITION_SOURCE = (
    "(1)(a)(1)(a) First definition.\n(b) Other first definition.\n"
    "(2) Relative means one of the following conditions:\n"
    "(a) The individual is related to the taxpayer.\n"
    "(b) The individual is a lineal descendant.\n"
    "(c) An entity is owned by the individual.\n"
)


@pytest.mark.parametrize(
    ("source", "citation", "source_sha256", "parent_start", "children"),
    [
        (
            IA_NET_INCOME_SOURCE,
            "us-ia/statute/422.7",
            "abddb6100fed80fb6834f1ff9f4ab8d475ee35c0f04ec490034a7a781a931bc6",
            15381,
            [(15467, "a"), (15596, "b"), (15910, "c")],
        ),
        (
            IA_CORPORATE_TAX_SOURCE,
            "us-ia/statute/422.33",
            "b95294c12c06d883bfd8857e789981940be9970617886ce6969986d186acab21",
            21036,
            [(21317, "a"), (21440, "b")],
        ),
    ],
    ids=["relative-sixth-definition", "research-credit-third-condition"],
)
def test_colon_owned_deep_first_child_keeps_its_numeric_parent(
    source, citation, source_sha256, parent_start, children
):
    assert hashlib.sha256(source.encode()).hexdigest() == source_sha256
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    parent = next(branch for branch in branches if branch.start == parent_start)
    for start, label in children:
        child = next(
            branch
            for branch in branches
            if branch.start == start and branch.label == f"({label})"
        )
        assert child.path == (*parent.path, label)
        assert parent.start <= child.start < child.end <= parent.end
        assert source[child.start : child.end].strip() == child.text
        assert sc._source_branch_direct_text(child, branches=branches)


def test_detached_first_child_retains_every_explicit_attached_ancestor():
    branches = sc.recognize_source_structure(
        DEEP_DEFINITION_SOURCE, corpus_citation_path="us-ia/statute/example"
    )
    paths = [branch.path for branch in branches]
    assert len(paths) == len(set(paths))
    for label in ("a", "b", "c"):
        assert ("1", "a", "2", label) in paths


@pytest.mark.parametrize("chapeau", ["conditions.", "conditions:\n"])
def test_detached_first_child_extension_preserves_chapeau_boundaries(chapeau):
    source = DEEP_DEFINITION_SOURCE.replace("conditions:", chapeau)
    paths = {
        branch.path
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path="us-ia/statute/example"
        )
    }
    assert ("1", "a", "2", "a") not in paths


def test_detached_first_child_extension_preserves_explicit_citation_chapeaux():
    source = DEEP_DEFINITION_SOURCE.replace(
        "Relative means one of the following conditions:",
        "See the following subdivisions:",
    ).replace(
        "(a) The individual is related to the taxpayer.\n"
        "(b) The individual is a lineal descendant.\n"
        "(c) An entity is owned by the individual.",
        "(a) and\n(b) of this section.",
    )
    paths = {
        branch.path
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path="us-ia/statute/example"
        )
    }
    assert ("1", "a", "2", "a") not in paths


def test_ordinary_profile_child_keeps_its_existing_numeric_owner():
    source = (
        "(1) The following conditions apply:\n(a) First condition.\n"
        "(b) Second condition.\n(2) The next root."
    )
    paths = {
        branch.path
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path="us-ia/statute/example"
        )
    }
    assert paths == {("1",), ("1", "a"), ("1", "b"), ("2",)}


@settings(max_examples=40, deadline=None)
@given(
    definition=st.integers(min_value=1, max_value=5),
    separator=st.sampled_from(["\n", "\r\n", "\n  ", "\r\n  "]),
)
def test_colon_owned_first_children_preserve_active_numeric_ancestry(
    definition, separator
):
    source = separator.join(
        [
            f"(1)(a)({definition})(a) First definition.",
            "(b) Other first definition.",
            f"({definition + 1}) Relative means the following conditions:",
            "(a) The individual is related to the taxpayer.",
            "(b) The individual is a lineal descendant.",
            "(c) An entity is owned by the individual.",
            f"({definition + 2}) The next definition.",
        ]
    )
    branches = sc.recognize_source_structure(
        source, corpus_citation_path="us-ia/statute/example"
    )
    paths = [branch.path for branch in branches]
    assert len(paths) == len(set(paths))
    by_path = {branch.path: branch for branch in branches}
    parent = ("1", "a", str(definition + 1))
    assert parent in by_path
    for label in ("a", "b", "c"):
        assert (*parent, label) in by_path
        child = by_path[(*parent, label)]
        assert by_path[parent].start <= child.start < child.end <= by_path[parent].end
        assert source[child.start : child.end].strip() == child.text
    assert ("1", "a", str(definition + 2)) in by_path
