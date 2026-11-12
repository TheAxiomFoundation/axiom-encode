"""Conjunctions separate operative items but also occur in legal references."""

from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness
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
        "for necessary medical care",
        "to pay necessary expenses",
        "with a separate permission",
        "that meets a separate condition",
        "which meets a separate condition",
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


@pytest.mark.parametrize("joiner", JOINERS)
def test_conjoined_capital_letter_requires_an_operative_predecessor(joiner):
    source = outline("(i) Qualifying categories" + joiner + "(B) Isolated category.")
    assert not any(branch.start == source.index("(B)") for branch in structural(source))


@pytest.mark.parametrize("joiner", JOINERS)
def test_later_reference_cannot_prove_an_operative_sibling(joiner):
    source = outline(
        "(i) A person described in clause (i)"
        + joiner
        + "(ii) must comply. See clause: (iii) for scope."
    )
    assert not any(
        branch.start in (source.index("(ii)"), source.index("(iii)"))
        for branch in structural(source)
    )


@pytest.mark.parametrize("joiner", JOINERS)
def test_previous_under_reference_does_not_own_next_condition(joiner):
    source = outline(
        "(i) made under clause (i)"
        + joiner
        + "(ii) for necessary medical care; (iii) for emergency travel."
    )
    assert any(
        branch.path == ("g", "ii") and branch.start == source.index("(ii)")
        for branch in structural(source)
    )


@pytest.mark.parametrize("joiner", JOINERS)
def test_conjoined_fused_children_keep_their_source_boundaries(joiner):
    source = outline("(i)(A) First condition" + joiner + "(ii)(A) Second condition.")
    assert [
        (branch.path, branch.start)
        for branch in structural(source)
        if len(branch.path) > 1
    ] == [
        (("g", "i"), source.index("(i)(A)")),
        (("g", "i", "a"), source.index("(i)(A)")),
        (("g", "ii"), source.index("(ii)(A)")),
        (("g", "ii", "a"), source.index("(ii)(A)")),
    ]


@pytest.mark.parametrize(
    "fixture,path,offset",
    [
        ("a-9", ("e", "1"), 38129),
        ("a-9", ("j", "1"), 46071),
        ("a-9", ("m", "1"), 56647),
        ("a-9", ("m", "1", "a"), 56776),
        ("a-9", ("m", "1", "b"), 56966),
        ("a-9", ("m", "2"), 57067),
        ("b-12", ("a", "1"), 43),
        ("b-12", ("a", "1", "a"), 136),
        ("b-12", ("a", "1", "b"), 253),
        ("b-12", ("a", "2"), 644),
        ("b-12", ("a", "2", "a"), 780),
        ("b-12", ("a", "2", "b"), 921),
        ("b-12", ("a", "3"), 1156),
        ("b-12", ("a", "4"), 1514),
        ("b-12", ("j", "1"), 25614),
    ],
)
def test_direct_root_chapeau_preserves_printed_fixture_ancestry(fixture, path, offset):
    source = (
        Path(__file__).parent
        / "fixtures"
        / "editorial_source_ancestry"
        / f"{fixture}-source.txt"
    ).read_text()
    branches = structural(source)
    matching = [
        branch for branch in branches if branch.path == path and branch.start == offset
    ]
    assert len(matching) == 1
    assert matching[0].text.startswith(source[offset : offset + 30])


ROOT_CHAPEAUX = (
    "For purposes of this chapter— ",
    "As used in this section— ",
    "For purposes of this subsection(a)(2): ",
)
REFERENCE_COMMANDS = (
    "See",
    "Compare",
    "Refer to",
    "Subject to",
    "Under",
    "Pursuant to",
    "According to",
    "In accordance with",
)


@pytest.mark.parametrize("chapeau", ROOT_CHAPEAUX)
def test_direct_root_chapeau_admits_first_numeric_child(chapeau):
    source = outline(chapeau + "(1) eligible persons; or (2) eligible entities.")
    assert [
        (branch.path, branch.start)
        for branch in structural(source)
        if len(branch.path) > 1
    ] == [
        (("g", "1"), source.index("(1) eligible")),
        (("g", "2"), source.index("(2) eligible")),
    ]


@pytest.mark.parametrize("command", REFERENCE_COMMANDS)
def test_explicit_reference_command_prevents_root_chapeau_override(command):
    source = outline(
        command + " the rule for purposes of this chapter— (1) of this section."
    )
    assert not any(branch.start == source.index("(1)") for branch in structural(source))


@pytest.mark.parametrize(
    "command", [command for command in REFERENCE_COMMANDS if " " in command]
)
def test_wrapped_reference_command_prevents_root_chapeau_override(command):
    source = outline(
        command.replace(" ", "\n")
        + " the rule for purposes of this chapter— (1) of this section."
    )
    assert not any(branch.start == source.index("(1)") for branch in structural(source))


@settings(max_examples=60, deadline=None)
@given(
    chapeau=st.sampled_from(ROOT_CHAPEAUX),
    command=st.sampled_from(REFERENCE_COMMANDS),
    space=st.sampled_from((" ", "\t", "  ", "\n", "\u00a0")),
)
def test_root_chapeau_never_overrides_an_explicit_reference(chapeau, command, space):
    source = outline(
        space.join(command.split())
        + space
        + "the rule "
        + chapeau
        + "(1) of this section for scope."
    )
    assert not any(branch.start == source.index("(1)") for branch in structural(source))


@pytest.mark.parametrize(
    "body",
    [
        "See paragraph: (1) of this section for scope.",
        "The person described in paragraph: (1) must comply.",
    ],
)
def test_same_root_numeric_references_keep_default_guard(body):
    source = outline(body)
    assert not any(branch.start == source.index("(1)") for branch in structural(source))


def test_numeric_line_cannot_borrow_earlier_root_chapeau_proof():
    source = outline("") + (
        "\n(1) Operative item. For purposes of this section— (1) of this section."
    )
    assert not any(
        branch.start == source.rindex("(1)") for branch in structural(source)
    )


@pytest.mark.parametrize(
    "body,first_path",
    [
        (
            "(g)(1) Operative item. For purposes of this section— (1) of this section.",
            ("g", "1"),
        ),
        (
            "(g)(i) Operative item. For purposes of this section— (1) of this section.",
            ("g", "i"),
        ),
        (
            "(g) First: (1) Operative item. For purposes of this section— "
            "(1) of this section.",
            ("g", "1"),
        ),
        (
            "(g) For purposes of this chapter— (1) employees qualify. "
            "For purposes of this section— (1) of this section.",
            ("g", "1"),
        ),
    ],
)
def test_existing_or_restored_child_prevents_later_chapeau_reference(body, first_path):
    prefix = "\n\n".join(
        f"({chr(index)}) Duty {chr(index)}." for index in range(ord("a"), ord("g"))
    )
    source = prefix + "\n\n" + body
    branches = structural(source)
    first_start = (
        source.index("(g)(") if body.startswith("(g)(") else source.index("(1)")
    )
    assert any(
        branch.path == first_path and branch.start == first_start for branch in branches
    )
    assert not any(branch.start == source.rindex("(1)") for branch in branches)


@settings(max_examples=30, deadline=None)
@given(count=st.integers(min_value=2, max_value=5))
def test_one_root_can_restore_only_one_first_numeric_child(count):
    source = outline(
        "For purposes of this chapter— (1) employees qualify."
        + " For purposes of this section— (1) of this section." * (count - 1)
    )
    assert [
        (branch.path, branch.start)
        for branch in structural(source)
        if len(branch.path) > 1
    ] == [(("g", "1"), source.index("(1)"))]


def test_misclassified_roman_root_cannot_borrow_an_earlier_root():
    # Root-level l versus Roman l remains the separate hierarchy problem.
    source = (
        Path(__file__).parent
        / "fixtures"
        / "editorial_source_ancestry"
        / "a-9-source.txt"
    ).read_text()
    first = source.index("(1) Agreement with respect to certain employees")
    source = (
        source[:first]
        + "For purposes of this section— "
        + source[first : source.index("\n(m)")]
    )
    target = source.index("(1) Agreement with respect to certain employees")
    assert not any(branch.start == target for branch in structural(source))


@pytest.mark.parametrize(
    "body,resolutions",
    [
        ("For purposes of this section— (1) eligible persons.", 2),
        ("See the rule for purposes of this section— (1) of this section.", 1),
        ("(i) first; or (ii) second.", 1),
    ],
)
def test_root_chapeau_uses_bounded_resolution_without_recursive_recognition(
    body, resolutions
):
    # Spies wrap the actual Axiom functions; no alternate evaluator is used.
    with (
        patch.object(
            source_completeness,
            "_resolve_parenthesized_legal_outline_markers",
            wraps=source_completeness._resolve_parenthesized_legal_outline_markers,
        ) as resolve,
        patch.object(
            source_completeness,
            "recognize_source_structure",
            wraps=source_completeness.recognize_source_structure,
        ) as recognize,
    ):
        source_completeness.recognize_source_structure(outline(body))
    assert recognize.call_count == 1
    assert resolve.call_count == resolutions


@pytest.mark.parametrize(
    "tail",
    [
        " of this section must apply.",
        " in this section must apply.",
        " under this section must apply.",
        " and (2) of this section must apply.",
        " or (2) of this section must apply.",
        " and/or (2) of this section must apply.",
        " and/\n or (2) of this section must apply.",
        " or(2) of this section must apply.",
        ", and (2)(a) of this subsection must apply.",
        "\n and\n (2)\n of\n this\n section must apply.",
        "\u00a0or\u00a0(2)\u00a0under\u00a0this\u00a0section must apply.",
        " of subsection(a) must apply.",
        " of such subsection(a) must apply.",
        " in any section must apply.",
        " under those sections must apply.",
        " of these subsections must apply.",
        " of each section must apply.",
        " in every section must apply.",
        " of a section must apply.",
        " under an article must apply.",
        " of said section must apply.",
        " of any such subsection(a) must apply.",
        " of each such section must apply.",
        " under any other section must apply.",
        " of such a section must apply.",
        " of either paragraph must apply.",
        " under both subsections must apply.",
        " of another section must apply.",
        " of all sections must apply.",
        " of some section must apply.",
        " of the same paragraph must apply.",
        " under the preceding subsection must apply.",
        " in the following section must apply.",
        " of this Act must apply.",
        " under this Code must apply.",
        " of the same sentence must apply.",
        " in this Schedule must apply.",
    ],
)
def test_reference_tail_cannot_become_a_new_first_numeric_child(tail):
    source = outline("For purposes of this chapter— (1)" + tail)
    assert not any(branch.start == source.index("(1)") for branch in structural(source))


@pytest.mark.parametrize(
    "body",
    [
        "eligible persons",
        "eligible persons.",
        "must meet a condition.",
        "is eligible.",
        "has a separate permission.",
        "under an agreement is eligible.",
        "under any such agreement is eligible.",
        "in a qualifying role is eligible.",
        "in such a qualifying role is eligible.",
        "of a specified type is eligible.",
        "under both agreements remains eligible.",
        "of another specified type is eligible.",
        "in some time period is eligible.",
    ],
)
def test_operative_first_child_keeps_its_eof_or_predicate_body(body):
    source = outline("For purposes of this chapter— (1) " + body)
    assert [
        (branch.path, branch.start)
        for branch in structural(source)
        if len(branch.path) > 1
    ] == [(("g", "1"), source.index("(1)"))]


@settings(max_examples=80, deadline=None)
@given(
    count=st.integers(min_value=1, max_value=8),
    conjunction=st.sampled_from(("and", "or", "and/or")),
    space=st.sampled_from((" ", "\t", "\n", "\u00a0")),
    qualifier=st.sampled_from(("of", "in", "under")),
    determiners=st.lists(
        st.sampled_from(
            (
                "this",
                "that",
                "the",
                "these",
                "those",
                "such",
                "any",
                "each",
                "every",
                "a",
                "an",
                "said",
                "other",
                "either",
                "both",
                "all",
                "another",
                "some",
                "same",
                "preceding",
                "following",
            )
        ),
        min_size=0,
        max_size=4,
    ),
    unit=st.sampled_from(
        (
            "section",
            "subsection",
            "paragraph",
            "article",
            "Act",
            "Code",
            "sentence",
            "Schedule",
        )
    ),
)
def test_generated_reference_tails_never_restore_an_outline_item(
    count, conjunction, space, qualifier, determiners, unit
):
    chain = (space + conjunction + space).join(
        f"({number})" for number in range(1, count + 1)
    )
    source = outline(
        "For purposes of this chapter— "
        + chain
        + space
        + qualifier
        + space
        + space.join(determiners)
        + space
        + unit
        + " must apply."
    )
    assert not any(branch.start == source.index("(1)") for branch in structural(source))


@pytest.mark.parametrize("count", [4, 32, 256])
@pytest.mark.parametrize("qualifies", [False, True])
def test_reference_tail_index_inspects_disjoint_windows(count, qualifies):
    source = "(1)" + " and (1)" * (count - 1)
    source += (
        " of " + "any such " * count + "section" if qualifies else " ordinary body"
    )
    with (
        patch.object(
            source_completeness,
            "_ROOT_CHAPEAU_TAIL_LABEL",
            wraps=source_completeness._ROOT_CHAPEAU_TAIL_LABEL,
        ) as labels,
        patch.object(
            source_completeness,
            "_ROOT_CHAPEAU_TAIL_JOINER",
            wraps=source_completeness._ROOT_CHAPEAU_TAIL_JOINER,
        ) as joiner,
        patch.object(
            source_completeness,
            "_ROOT_CHAPEAU_TAIL_QUALIFIER",
            wraps=source_completeness._ROOT_CHAPEAU_TAIL_QUALIFIER,
        ) as qualifier,
    ):
        blocked = source_completeness._root_chapeau_reference_tail_starts(source)
    expected = {index * 8 for index in range(count)} if qualifies else set()
    assert blocked == expected
    assert labels.finditer.call_count == 1
    assert joiner.fullmatch.call_count == count - 1
    assert qualifier.match.call_count == count
    for calls in (joiner.fullmatch.call_args_list, qualifier.match.call_args_list):
        assert sum(call.args[2] - call.args[1] for call in calls) <= len(source)
