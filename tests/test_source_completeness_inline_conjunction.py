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


# Verbatim paragraph (2) from replay fixture37154591758, 26 USC911(f).
# Only the preceding root is a synthetic scaffold; C has a printed heading.
# Paragraph SHA256: 6f0c6b7e98ccc15bd8d01ce47d3a86ebcfee8e3b53290151631d513c73542242
TAX_SPECIAL_RULES_SOURCE = (
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


HEADING_TITLES = (
    "General rule",
    "GENERAL RULES",
    "Special rules",
    "Definition",
    "Definitions",
    "In general",
)


def headed_source(
    title="Special rules", body="The first rule applies.", *, sibling=True
):
    return (
        "(1) Preceding root.\n\n(2) "
        + title
        + " (A) "
        + body
        + ("\n(B) The second rule applies." if sibling else "")
    )


@pytest.mark.parametrize("title", HEADING_TITLES)
def test_strong_heading_recovers_first_A_and_its_exact_boundary(title):
    source = headed_source(title)
    branches = structural(source)
    first = next(branch for branch in branches if branch.path == ("2", "a"))
    assert first.start == source.index("(A)")
    assert first.end == source.index("(B)")
    assert first.text == source[first.start : first.end].strip()
    assert any(branch.path == ("2", "b") for branch in branches)


def test_heading_recovers_printed_A9_church_definition():
    source = (
        Path(__file__).parent
        / "fixtures"
        / "editorial_source_ancestry"
        / "a-9-source.txt"
    ).read_text()
    first = next(
        branch for branch in structural(source) if branch.path == ("w", "3", "a")
    )
    assert first.start == 73568
    assert first.end == 73832
    assert first.text == source[73568:73832].strip()


def test_tax_heading_recovers_A_B_C_and_their_printed_boundaries():
    source = TAX_SPECIAL_RULES_SOURCE
    branches = structural(source)
    for path, text in [
        (("2", "a"), "(A) Regular tax"),
        (("2", "a", "i"), "(i) the taxpayer’s net capital gain"),
        (("2", "b"), "(B) Alternative minimum tax"),
        (("2", "b", "i"), "(i) the rules of subparagraph"),
        (("2", "b", "ii"), "(ii) the reference"),
    ]:
        assert any(
            branch.path == path and branch.start == source.index(text)
            for branch in branches
        )
    a = next(branch for branch in branches if branch.path == ("2", "a"))
    first = next(branch for branch in branches if branch.path == ("2", "a", "i"))
    assert a.end == first.end == source.index("(B) Alternative minimum tax")
    c_start = source.index("(C) Definitions")
    c_match = next(
        match
        for match in source_completeness._INLINE_PARENTHESIZED_OUTLINE_MARKER.finditer(
            source
        )
        if match.start("marker") == c_start
    )
    assert source_completeness._inline_outline_marker_has_reference_context(
        source, c_match
    )
    matches = [branch for branch in branches if branch.start == c_start]
    assert len(matches) == 1
    c = matches[0]
    assert c.path == ("2", "c") and c.end == len(source)
    assert c.text == source[c_start:].strip()
    second = next(branch for branch in branches if branch.path == ("2", "b", "ii"))
    b = next(branch for branch in branches if branch.path == ("2", "b"))
    assert b.end == second.end == c_start
    assert b.text == source[b.start : c_start].strip()
    assert second.text == source[second.start : c_start].strip()
    assert "(C) Definitions" not in second.text


@pytest.mark.parametrize("title", ["Applicability of", "Meaning of"])
def test_arbitrary_word_heading_cannot_turn_a_reference_into_first_A(title):
    # These are red controls against the rejected arbitrary-word heading rule.
    source = headed_source(title, "is the subject of this cross-reference.")
    assert not any(branch.start == source.index("(A)") for branch in structural(source))


@pytest.mark.parametrize("command", REFERENCE_COMMANDS)
def test_reference_command_heading_never_restores_first_A(command):
    source = headed_source(command + " statute")
    assert not any(branch.start == source.index("(A)") for branch in structural(source))


@pytest.mark.parametrize(
    "tail",
    [
        "of this section.",
        "in any such subsection.",
        "under the same Act.",
        "and\n(B) of this section.",
        "; or\n(B) under both subsections.",
    ],
)
def test_heading_does_not_restore_a_reference_tail(tail):
    source = headed_source(body=tail)
    assert not any(branch.start == source.index("(A)") for branch in structural(source))


@pytest.mark.parametrize(
    "body",
    [
        "under an agreement the claimant receives payment.",
        "in a supervisory role the claimant signs the form.",
        "in the following year the claimant receives payment.",
    ],
)
def test_heading_reference_veto_preserves_operative_bodies(body):
    source = headed_source(body=body)
    assert any(
        branch.path == ("2", "a") and branch.start == source.index("(A)")
        for branch in structural(source)
    )


@pytest.mark.parametrize(
    "source",
    [
        headed_source(sibling=False),
        headed_source(sibling=False) + "\n(3) Different parent.\n(B) Other rule.",
        "(1) Special rules (A) First rule.\n(B) Second rule.",
        headed_source().replace("(A)", "(a)"),
        headed_source().replace("(A)", "(A)(i)"),
        "(1) Preceding root.\n\n(2)(i) Special rules (A) Reference.\n(B) Other rule.",
    ],
)
def test_heading_cannot_borrow_a_parent_or_sibling(source):
    start = source.index("(A)") if "(A)" in source else source.index("(a)")
    assert not any(branch.start == start for branch in structural(source))


@pytest.mark.parametrize("offset", [54463, 63044, 70110, 78469])
def test_other_unproven_A9_heading_parents_remain_held(offset):
    source = (
        Path(__file__).parent
        / "fixtures"
        / "editorial_source_ancestry"
        / "a-9-source.txt"
    ).read_text()
    assert source[offset:].startswith("(A)")
    assert not any(branch.start == offset for branch in structural(source))


@settings(max_examples=60, deadline=None)
@given(title=st.sampled_from(HEADING_TITLES), space=st.sampled_from((" ", "  ", "\t")))
def test_generated_strong_headings_keep_parent_and_sibling_boundaries(title, space):
    source = (
        headed_source(title.replace(" ", space))
        .replace("(2) ", "(2)" + space)
        .replace(" (A)", space + "(A)")
    )
    branches = structural(source)
    first = next(branch for branch in branches if branch.path == ("2", "a"))
    assert (first.start, first.end) == (source.index("(A)"), source.index("(B)"))
    assert any(
        branch.path == ("2", "b") and branch.start == source.index("(B)")
        for branch in branches
    )


@settings(max_examples=60, deadline=None)
@given(
    count=st.integers(min_value=1, max_value=6),
    conjunction=st.sampled_from(("and", "or", "; or", ", and")),
    space=st.sampled_from((" ", "\t", "\n", "\u00a0")),
    qualifier=st.sampled_from(("of", "in", "under")),
    determiner=st.sampled_from(
        ("this", "any such", "the same", "either", "both", "all")
    ),
    unit=st.sampled_from(
        ("section", "subsection", "paragraph", "Act", "Code", "sentence", "Schedule")
    ),
)
def test_generated_first_A_reference_chains_stay_references(
    count, conjunction, space, qualifier, determiner, unit
):
    chain = (space + conjunction + space).join(
        "(" + chr(ord("A") + i) + ")" for i in range(count)
    )
    source = (
        "(1) Preceding root.\n\n(2) Definitions "
        + chain
        + space
        + qualifier
        + space
        + determiner
        + space
        + unit
        + ".\n(B) Independent rule."
    )
    assert not any(branch.start == source.index("(A)") for branch in structural(source))


def test_mixed_independent_numeric_and_heading_restorations_share_one_final_pass():
    source = outline("For purposes of this chapter— (1) eligible persons.")
    source += (
        "\n\n(h) Another root.\n(1) Special rules (A) First rule.\n(B) Second rule."
    )
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
        branches = source_completeness.recognize_source_structure(
            source, corpus_citation_path=CITATION
        )
    assert recognize.call_count == 1
    assert resolve.call_count == 2
    assert any(branch.path == ("g", "1") for branch in branches)
    assert any(branch.path == ("h", "1", "a") for branch in branches)


def test_new_numeric_restoration_cannot_supply_heading_parent_proof():
    source = outline(
        "For purposes of this chapter— (1) Special rules (A) First rule.\n(B) Second rule."
    )
    assert any(branch.path == ("g", "1") for branch in structural(source))
    assert not any(branch.start == source.index("(A)") for branch in structural(source))


@pytest.mark.parametrize("count", [4, 32, 256])
@pytest.mark.parametrize("qualifies", [False, True])
def test_first_A_reference_tail_index_uses_disjoint_windows(count, qualifies):
    source = "(A)" + " and (A)" * (count - 1)
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
        blocked = source_completeness._headed_first_alpha_reference_tail_starts(source)
    assert blocked == ({index * 8 for index in range(count)} if qualifies else set())
    assert labels.finditer.call_count == 1
    assert joiner.fullmatch.call_count == count - 1
    assert qualifier.match.call_count == count
    for calls in (joiner.fullmatch.call_args_list, qualifier.match.call_args_list):
        assert sum(call.args[2] - call.args[1] for call in calls) <= len(source)


def test_heading_match_is_anchored_once_per_existing_numeric_parent():
    source = "\n\n".join(
        f"({number}) Special rules "
        + " " * 256
        + "(A) of this section. "
        + " and (A)" * 32
        + "\n(B) Independent rule."
        for number in range(1, 17)
    )
    with patch.object(
        source_completeness,
        "_HEADED_FIRST_ALPHA_OUTLINE_MARKER",
        wraps=source_completeness._HEADED_FIRST_ALPHA_OUTLINE_MARKER,
    ) as heading:
        structural(source)
    starts = [call.args[1] for call in heading.match.call_args_list]
    assert len(starts) == len(set(starts)) == 16


def completed_heading_source(
    body="The second rule operates under paragraph (1).",
    tail="Definitions Terms used retain their meanings.",
):
    return (
        "(1) First root.\n\n(2) General rules (A) First rule applies. (B) "
        + body
        + " (C) "
        + tail
        + "\n\n(3) Final root."
    )


@pytest.mark.parametrize(
    "title",
    (
        "General rule",
        "General rules",
        "Special rules",
        "Definition",
        "Definitions",
        "In general",
    ),
)
def test_completed_sentence_recovers_C_and_only_its_printed_span(title):
    source = completed_heading_source(tail=title + " Terms retain meanings.")
    branches = structural(source)
    start = source.index("(C)")
    matches = [b for b in branches if b.start == start]
    assert len(matches) == 1
    c = matches[0]
    assert c.path == ("2", "c") and c.end == source.index("(3)")
    assert c.text == source[c.start : c.end].strip()
    b = next(b for b in branches if b.path == ("2", "b"))
    assert b.end == start and "(C)" not in b.text


@pytest.mark.parametrize(
    ("name", "source"),
    [
        (
            "supported-curly",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule quotes “Text under paragraph (1). (C) Definitions Terms retain meanings.” End.\n\n(3) Final root.",
        ),
        (
            "escaped-ascii-quotes",
            '(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule quotes "Text \\"under paragraph (1). (C) Definitions Terms retain meanings. \\"more" End.\n\n(3) Final root.',
        ),
        (
            "backtick-inline",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule quotes `Text under paragraph (1). (C) Definitions Terms retain meanings.` End.\n\n(3) Final root.",
        ),
        (
            "backtick-fence",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule quotes ```Text under paragraph (1). (C) Definitions Terms retain meanings.``` End.\n\n(3) Final root.",
        ),
        (
            "unsupported-single-guillemets",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule quotes ‹Text under paragraph (1). (C) Definitions Terms retain meanings.› End.\n\n(3) Final root.",
        ),
        (
            "balanced-editorial-brackets",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule includes [Text under paragraph (1). (C) Definitions Terms retain meanings.] End.\n\n(3) Final root.",
        ),
        (
            "lowercase-see-last-sentence",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The second rule applies. see paragraph (1). (C) Definitions Terms used in this paragraph retain their meanings.\n\n(3) Final root.",
        ),
        (
            "upper-see-last-sentence",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The second rule applies. See paragraph (1). (C) Definitions Terms used in this paragraph retain their meanings.\n\n(3) Final root.",
        ),
        (
            "right-three-reference-chain",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule operates under paragraph (1). (C) Definitions and (D) or (E) of this section remain controlling.\n\n(3) Final root.",
        ),
        (
            "right-four-reference-chain",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule operates under paragraph (1). (C) Definitions and (D), (E), and (F) under this section remain controlling.\n\n(3) Final root.",
        ),
        (
            "repeated-B-identity",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) First B. (B) The rule operates under paragraph (1). (C) Definitions Terms used in this paragraph retain their meanings.\n\n(3) Final root.",
        ),
        (
            "repeated-numeric-parent-identity",
            "(1) First root.\n\n(2) Earlier duplicate.\n\n(2) General rules (A) First rule applies. (B) The rule operates under paragraph (1). (C) Definitions Terms used in this paragraph retain their meanings.\n\n(3) Final root.",
        ),
        (
            "closed-parent-next-root",
            "(1) First root.\n\n(2) General rules (A) Rule A. (B) Rule B.\n\n(3) Text operates under paragraph (1). (C) Definitions Terms retain meanings.\n\n(4) Last root.",
        ),
        (
            "new-C-cannot-prove-D",
            "(1) First root.\n\n(2) General rules (A) First rule applies. (B) The rule operates under paragraph (1). (C) Definitions First new rule operates under paragraph (1). (D) Definitions Terms retain meanings.\n\n(3) Final root.",
        ),
    ],
)
def test_later_heading_red_reference_contexts_and_ambiguous_proof_stay_held(
    name, source
):
    # These include the ten false admissions of the rejected V3 diagnostic.
    label = "(D)" if name == "new-C-cannot-prove-D" else "(C)"
    assert not any(b.start == source.index(label) for b in structural(source))


@pytest.mark.parametrize(
    "command",
    [
        "see",
        "compare",
        "refer to",
        "under",
        "subject to",
        "pursuant to",
        "according to",
        "in accordance with",
        "as provided in",
    ],
)
def test_later_heading_final_sentence_commands_are_case_insensitive(command):
    source = completed_heading_source(
        "The second rule applies. " + command + " paragraph (1)."
    )
    assert not any(b.start == source.index("(C)") for b in structural(source))


@pytest.mark.parametrize(
    "body",
    [
        "The taxpayer’s rule operates under paragraph (1).",
        "The U.S. rule operates under paragraph (1).",
        "The rule, e.g. this rule, operates under paragraph (1).",
    ],
)
def test_later_heading_preserves_possessives_and_nonboundary_abbreviations(body):
    source = completed_heading_source(body)
    assert any(
        b.path == ("2", "c") and b.start == source.index("(C)")
        for b in structural(source)
    )


@pytest.mark.parametrize(
    ("raw_length", "admitted"), [(511, True), (512, True), (513, False)]
)
def test_later_heading_previous_leaf_window_is_bounded(raw_length, admitted):
    prefix, suffix = "The rule ", " operates under paragraph (1)."
    body = prefix + "x" * (raw_length - 2 - len(prefix) - len(suffix)) + suffix
    source = completed_heading_source(body)
    assert source.index("(C)") - (source.index("(B)") + 3) == raw_length
    assert any(b.start == source.index("(C)") for b in structural(source)) == admitted


@settings(max_examples=25, deadline=None)
@given(
    count=st.integers(2, 20),
    joiner=st.sampled_from((" and ", " or ", ", ", "; ", " and/or ")),
    qualifier=st.sampled_from(
        ("of this section", "under that paragraph", "in the same Act")
    ),
)
def test_later_heading_complete_coordinate_reference_chains_are_never_proof(
    count, joiner, qualifier
):
    chain = joiner.join("(" + chr(ord("D") + i) + ")" for i in range(count))
    source = completed_heading_source(
        tail="Definitions" + joiner + chain + " " + qualifier + " remain controlling."
    )
    assert not any(b.start == source.index("(C)") for b in structural(source))


def test_new_C_does_not_supply_the_baseline_sibling_for_D():
    source = completed_heading_source(
        tail="Definitions The rule operates under paragraph (1). (D) Definitions Terms retain meanings."
    )
    branches = structural(source)
    assert any(b.path == ("2", "c") for b in branches)
    assert not any(b.start == source.index("(D)") for b in branches)


def test_an_existing_baseline_C_can_independently_prove_D():
    source = (
        completed_heading_source()
        .replace(
            "(B) The second rule operates",
            "(B) Second rule. (C) The third rule operates",
        )
        .replace(" (C) Definitions", " (D) Definitions")
    )
    branches = structural(source)
    assert any(
        b.path == ("2", "d") and b.start == source.index("(D)") for b in branches
    )


@pytest.mark.parametrize(
    "source",
    [
        completed_heading_source().replace("(2) General rules (A)", "(2)(A)", 1),
        completed_heading_source().replace(
            "(2) General rules (A) First rule applies. (B)",
            "(2) General rules (A) First rule applies.\n\n(3) (B)",
            1,
        ),
    ],
)
def test_later_heading_cannot_borrow_shared_or_different_parent_coordinates(source):
    assert not any(b.start == source.index("(C)") for b in structural(source))


@pytest.mark.parametrize(
    ("source", "passes"),
    [
        ("(1) First root.\n\n(2) Second root.", 1),
        (headed_source(), 2),
        (completed_heading_source(), 3),
        (
            completed_heading_source().replace(
                "General rules (A)", "Existing list\n(A)"
            ),
            2,
        ),
    ],
)
def test_later_heading_keeps_the_bounded_internal_resolution_count(source, passes):
    with patch.object(
        source_completeness,
        "_resolve_parenthesized_legal_outline_markers",
        wraps=source_completeness._resolve_parenthesized_legal_outline_markers,
    ) as resolver:
        structural(source)
    assert resolver.call_count == passes


@pytest.mark.parametrize("count", [1, 64, 256])
def test_later_heading_chain_index_has_disjoint_bounded_windows(count):
    source = "(C) Definitions and (D) " * count + "of this section"
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
        blocked = source_completeness._later_heading_reference_tail_starts(source)
    assert len(blocked) == count
    assert labels.finditer.call_count == 1
    assert joiner.fullmatch.call_count == 2 * count - 1
    assert qualifier.match.call_count == 2 * count
    for calls in (joiner.fullmatch.call_args_list, qualifier.match.call_args_list):
        assert all(len(call.args) == 3 for call in calls)
        assert sum(call.args[2] - call.args[1] for call in calls) <= len(source)


def test_later_heading_never_recurses_to_the_public_recognizer():
    with patch.object(
        source_completeness,
        "recognize_source_structure",
        side_effect=AssertionError("recursive recognition"),
    ) as public:
        markers = source_completeness._qualified_parenthesized_legal_outline_markers(
            completed_heading_source()
        )
    assert any(m.path == ("2", "c") for m in markers)
    public.assert_not_called()


@pytest.mark.parametrize(
    ("body", "tail"),
    [
        (
            'The rule quotes "Text under paragraph (1).',
            'Definitions Terms retain meanings." End.',
        ),
        (
            "The rule quotes “Text under paragraph (1).",
            "Definitions Terms retain meanings.",
        ),
        (
            'The rule quotes "Text under paragraph (1).',
            "Definitions Terms retain meanings.",
        ),
        (
            "The rule includes [Text under paragraph (1).",
            "Definitions Terms retain meanings.",
        ),
        (
            "The rule includes (Text under paragraph (1).",
            "Definitions Terms retain meanings.) End.",
        ),
        (
            "The rule includes {Text under paragraph (1).",
            "Definitions Terms retain meanings.} End.",
        ),
        (
            "The rule quotes 〝Text under paragraph (1).",
            "Definitions Terms retain meanings.〞 End.",
        ),
        ('"Text under paragraph (1).".', "Definitions Terms retain meanings."),
    ],
)
def test_later_heading_uncertain_or_editorial_context_never_supplies_proof(body, tail):
    source = completed_heading_source(body, tail)
    assert not any(b.start == source.index("(C)") for b in structural(source))


@pytest.mark.parametrize(("opening", "closing"), [('"', '"'), ("[", "]")])
def test_later_heading_cannot_borrow_baseline_labels_inside_a_quote_or_group(
    opening, closing
):
    source = (
        "(1) First root.\n\n(2) General rules "
        + opening
        + "\n(A) Quoted first rule.\n(B) Quoted second rule."
        + closing
        + " The actual tail operates under paragraph (1). (C) Definitions Terms retain meanings."
        + "\n\n(3) Final root."
    )
    assert not any(b.start == source.index("(C)") for b in structural(source))


def completed_heading_with_uncertain_proof(role, opening, closing):
    tail = " The actual tail operates under paragraph (1). (C) Definitions Terms retain meanings.\n\n(3) Final root."
    if role == "parent":
        return (
            "(1) First root. "
            + opening
            + "\n\n(2) Quoted heading."
            + closing
            + "\n(A) First rule.\n(B) Second rule."
            + tail
        )
    if role == "first":
        return (
            "(1) First root.\n\n(2) General rules "
            + opening
            + "\n(A) Quoted first rule."
            + closing
            + "\n(B) Second rule."
            + tail
        )
    if role == "predecessor":
        return (
            "(1) First root.\n\n(2) General rules (A) First rule. "
            + opening
            + "\n(B) Quoted second rule."
            + closing
            + "\n(i) Intro rule.\n(ii) Last leaf."
            + tail
        )
    assert role == "leaf"
    return (
        "(1) First root.\n\n(2) General rules (A) First rule. (B) Second rule.\n(i) Intro rule. "
        + opening
        + "\n(ii) Quoted leaf."
        + closing
        + tail
    )


@pytest.mark.parametrize(
    "role,marker",
    [("parent", "(2)"), ("first", "(A)"), ("predecessor", "(B)"), ("leaf", "(ii)")],
)
@pytest.mark.parametrize("opening,closing", [('"', '"'), ("[", "]")])
def test_every_recovery_evidence_identity_must_be_outside_uncertain_context(
    role, marker, opening, closing
):
    operative = completed_heading_with_uncertain_proof(role, "", "")
    assert any(b.start == operative.index("(C)") for b in structural(operative))
    source = completed_heading_with_uncertain_proof(role, opening, closing)
    branches = structural(source)
    # Legacy baseline labels are preserved; only their use as NEW proof is refused.
    assert any(b.start == source.index(marker) for b in branches)
    assert not any(b.start == source.index("(C)") for b in branches)
