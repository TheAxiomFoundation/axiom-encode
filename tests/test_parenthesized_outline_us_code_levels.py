"""U.S. Code subclauses (I), items (aa) and subitems (AA) nest below clauses.

U.S. Code outlines run subsection (a), paragraph (1), subparagraph (A), clause
(i), subclause (I), item (aa) and subitem (AA). The CFR-style levels read (I)
as the letter after (H), (cc) as roman 200 and (aa) as no label, so the
subclauses of 42 U.S.C. 1396a(xx)(3)(A)(i) became phantom subparagraphs and
its items disappeared into the subclause text. The recognizer opens these
levels only on evidence from the outline: (I) opens subclauses under a clause
when (II) is the next label at its level, (aa) opens items under a subclause
when (bb) is next, and (AA) opens subitems under an item when (BB) is next.
Everything else reads as before, and inline references stay running text.
Triple letters such as (aaa) stay out of scope and read as before.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import recognize_source_structure

_FIXTURES = Path(__file__).parent / "fixtures/source_completeness"
_CITATION = "us/statute/42/1396a"
_XX_CITATION = "us/statute/42/1396a/xx"


def _paths(text: str, citation: str = _CITATION) -> list[str]:
    return [
        "/".join(branch.path)
        for branch in recognize_source_structure(text, corpus_citation_path=citation)
        if branch.kind in {"paragraph", "letter"}
    ]


def _items(*letters: str) -> str:
    return "".join(f"({letter * 2}) item {letter};\n" for letter in letters)


def _subclause_body(*lines: str) -> str:
    """Wrap ``lines`` as the children of clause (1)(A)(i), then close (A) and (1)."""

    return (
        "(1) In general.—\n(A) Rule.—\n(i) clause one—\n"
        + "".join(lines)
        + "(B) Next subparagraph.\n(2) Next paragraph.\n"
    )


# 42 U.S.C. 1396a(xx) as Pub. L. 119-21, sec. 71119, inserted it, one line per
# provision from the USLM text of the public law, and the lower-cased
# identifiers that the Office of the Law Revision Counsel gives each provision
# of the codified subsection (release 119-100). The first line is the heading
# of subsection (xx), which the body under it omits.
def test_42_usc_1396a_xx_parses_to_the_olrc_outline():
    lines = (_FIXTURES / "usc_42_1396a_xx_body.txt").read_text().splitlines(True)
    expected = (_FIXTURES / "usc_42_1396a_xx_olrc_paths.txt").read_text().split()
    body = "".join(lines[1:])

    paths = _paths(body, _XX_CITATION)

    assert paths == expected
    assert len(paths) == 111
    assert "3/b/ii/ii/bb" in paths
    assert [sum(path.count("/") == depth for path in paths) for depth in range(6)] == [
        11,
        25,
        30,
        24,
        19,
        2,
    ]


def test_42_usc_1396a_xx_3_files_subclauses_items_and_subitems():
    lines = (_FIXTURES / "usc_42_1396a_xx_body.txt").read_text().splitlines(True)
    body = "".join(lines[1:])

    paths = [path for path in _paths(body, _XX_CITATION) if path.startswith("3/")]

    assert paths == [
        "3/a",
        "3/a/i",
        "3/a/i/i",
        "3/a/i/ii",
        "3/a/i/ii/aa",
        "3/a/i/ii/bb",
        "3/a/i/ii/cc",
        "3/a/ii",
        "3/b",
        "3/b/i",
        "3/b/ii",
        "3/b/ii/i",
        "3/b/ii/ii",
        "3/b/ii/ii/aa",
        "3/b/ii/ii/bb",
        "3/b/ii/ii/bb/aa",
        "3/b/ii/ii/bb/bb",
        "3/b/ii/iii",
    ]


def test_h_i_j_stays_a_lettered_list():
    text = (
        "(1) Rule.\n"
        + "".join(f"({letter}) Duty {letter}.\n" for letter in "ABCDEFGH")
        + "(I) Duty I.\n(J) Duty J.\n(2) Next.\n"
    )

    assert _paths(text) == [
        "1",
        *[f"1/{letter}" for letter in "abcdefghij"],
        "2",
    ]


def test_subparagraph_i_after_a_clause_of_h_stays_lettered():
    # (I) after clause (ii) is subparagraph (I): its next label at its level
    # is (J), past its own clauses, not (II).
    text = (
        "(1) Rule.\n(G) g.\n(H) h—\n(i) one—\n(I) p;\n(II) q.\n(ii) two.\n"
        "(I) i—\n(i) one;\n(ii) two.\n(J) j.\n(2) Next.\n"
    )

    assert _paths(text) == [
        "1",
        "1/g",
        "1/h",
        "1/h/i",
        "1/h/i/i",
        "1/h/i/ii",
        "1/h/ii",
        "1/i",
        "1/i/i",
        "1/i/ii",
        "1/j",
        "2",
    ]


def test_aa_after_z_continues_the_lettered_list():
    text = (
        "".join(
            f"({letter}) Duty {letter}.\n" for letter in "abcdefghijklmnopqrstuvwxyz"
        )
        + "(aa) Duty aa.\n(bb) Duty bb.\n(cc) Duty cc.\n"
    )

    assert _paths(text) == [*"abcdefghijklmnopqrstuvwxyz", "aa", "bb", "cc"]


def test_ii_after_item_hh_is_a_clause_when_a_subclause_follows():
    # 42 U.S.C. 17231(k)(3)(D): items (aa) to (hh) end subclause (II), and
    # clause (ii) opens its own subclauses.
    text = _subclause_body(
        "(I) x;\n(II) y—\n",
        _items(*"abcdefgh"),
        "(ii) Priority—\n(I) p;\n(II) q.\n",
    )

    assert _paths(text) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/ii",
        *[f"1/a/i/ii/{letter * 2}" for letter in "abcdefgh"],
        "1/a/ii",
        "1/a/ii/i",
        "1/a/ii/ii",
        "1/b",
        "2",
    ]


def test_ii_after_item_hh_is_an_item_when_clause_ii_follows():
    # 42 U.S.C. 13503(b)(3)(E)(i)(V): item (ii) ends the items, and clause
    # (ii) follows it.
    text = _subclause_body(
        "(I) x;\n(II) y—\n",
        _items(*"abcdefgh"),
        "(ii) item i.\n(ii) Activities—\n(I) p;\n(II) q.\n",
    )

    assert _paths(text) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/ii",
        *[f"1/a/i/ii/{letter * 2}" for letter in "abcdefghi"],
        "1/a/ii",
        "1/a/ii/i",
        "1/a/ii/ii",
        "1/b",
        "2",
    ]


def test_ii_after_item_hh_is_an_item_when_jj_or_the_next_subclause_follows():
    jj = _subclause_body(
        "(I) x;\n(II) y—\n", _items(*"abcdefgh"), "(ii) item i;\n(jj) item j.\n"
    )
    subclause = _subclause_body(
        "(I) x;\n(II) y—\n", _items(*"abcdefgh"), "(ii) item i.\n(III) z.\n"
    )

    assert _paths(jj) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/ii",
        *[f"1/a/i/ii/{letter * 2}" for letter in "abcdefghij"],
        "1/b",
        "2",
    ]
    assert _paths(subclause) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/ii",
        *[f"1/a/i/ii/{letter * 2}" for letter in "abcdefghi"],
        "1/a/i/iii",
        "1/b",
        "2",
    ]


def test_cc_after_item_bb_is_an_item_not_roman_200():
    text = _subclause_body("(I) x—\n", _items(*"abc"), "(II) y.\n(ii) two.\n")

    assert _paths(text) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/i/aa",
        "1/a/i/i/bb",
        "1/a/i/i/cc",
        "1/a/i/ii",
        "1/a/ii",
        "1/b",
        "2",
    ]


def test_inline_references_to_subclauses_and_items_stay_running_text():
    text = _subclause_body(
        "(I) described in subclauses (I) through (VII) of subsection "
        "(a)(10)(A)(i); or\n",
        "(II) under items (aa) and (bb) of clause (ii): (aa) as such.\n",
    )

    branches = [
        branch
        for branch in recognize_source_structure(text, corpus_citation_path=_CITATION)
        if branch.kind in {"paragraph", "letter"}
    ]

    assert ["/".join(branch.path) for branch in branches] == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/ii",
        "1/b",
        "2",
    ]
    assert "subclauses (I) through (VII)" in branches[3].text
    assert "(a)(10)(A)(i)" in branches[3].text
    assert "items (aa) and (bb) of clause (ii): (aa) as such." in branches[4].text


def test_later_subparagraph_clause_or_paragraph_closes_items():
    text = (
        "(1) Rule.\n(A) Lead—\n(i) one—\n(I) first—\n(aa) x;\n(bb) y.\n"
        "(II) second—\n(aa) x;\n(bb) y.\n"
        "(ii) two—\n(I) p;\n(II) q—\n(aa) r;\n(bb) s.\n"
        "(B) b—\n(i) b1—\n(I) t;\n(II) u—\n(aa) v;\n(bb) w.\n"
        "(2) Next.\n"
    )

    assert _paths(text) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/i/aa",
        "1/a/i/i/bb",
        "1/a/i/ii",
        "1/a/i/ii/aa",
        "1/a/i/ii/bb",
        "1/a/ii",
        "1/a/ii/i",
        "1/a/ii/ii",
        "1/a/ii/ii/aa",
        "1/a/ii/ii/bb",
        "1/b",
        "1/b/i",
        "1/b/i/i",
        "1/b/i/ii",
        "1/b/i/ii/aa",
        "1/b/i/ii/bb",
        "2",
    ]


def test_attached_chains_open_subclauses_and_items():
    clause_and_subclause = (
        "(1) Rule.\n(A) Lead—\n(i)(I) first—\n(aa) x;\n(bb) y;\n(II) second;\n"
        "(ii) two.\n(B) b.\n(2) Next.\n"
    )
    subclause_and_item = _subclause_body(
        "(I) first;\n", "(II)(aa) x;\n", "(bb) y.\n", "(ii) two.\n"
    )

    assert _paths(clause_and_subclause) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/i/aa",
        "1/a/i/i/bb",
        "1/a/i/ii",
        "1/a/ii",
        "1/b",
        "2",
    ]
    assert _paths(subclause_and_item) == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/ii",
        "1/a/i/ii/aa",
        "1/a/i/ii/bb",
        "1/a/ii",
        "1/b",
        "2",
    ]


def test_subclause_without_ii_reads_as_before():
    # A lone (I) shows no subclause list, so it keeps the CFR-style reading:
    # the letter after (A), with (ii) under it.
    text = "(1) Rule.\n(A) Lead—\n(i) one—\n(I) only;\n(ii) two.\n(B) b.\n(2) Next.\n"

    assert _paths(text) == ["1", "1/a", "1/a/i", "1/i", "1/i/ii", "1/b", "2"]


def test_item_without_bb_stays_running_text():
    # 42 U.S.C. 1396a(a)(10)(A)(i)(II) holds one item, (aa), so no list shows.
    text = _subclause_body("(I) first;\n(II)\n(aa) only item;\n(III) third.\n")

    branches = [
        branch
        for branch in recognize_source_structure(text, corpus_citation_path=_CITATION)
        if branch.kind in {"paragraph", "letter"}
    ]

    assert ["/".join(branch.path) for branch in branches] == [
        "1",
        "1/a",
        "1/a/i",
        "1/a/i/i",
        "1/a/i/ii",
        "1/a/i/iii",
        "1/b",
        "2",
    ]
    assert "(aa) only item;" in branches[4].text


def test_triple_letters_stay_out_of_scope():
    text = (
        "".join(
            f"({letter}) Duty {letter}.\n" for letter in "abcdefghijklmnopqrstuvwxyz"
        )
        + "(aa) Duty aa.\n(bb) Duty bb.\n(aaa) Duty aaa.\n"
    )

    branches = [
        branch
        for branch in recognize_source_structure(text, corpus_citation_path=_CITATION)
        if branch.kind in {"paragraph", "letter"}
    ]

    assert ["/".join(branch.path) for branch in branches] == [
        *"abcdefghijklmnopqrstuvwxyz",
        "aa",
        "bb",
    ]
    assert "(aaa) Duty aaa." in branches[-1].text


def test_shared_label_helpers_keep_their_answers():
    assert not completeness_module._is_parenthesized_outline_label("aa")
    assert not completeness_module._is_parenthesized_outline_label("AA")
    assert completeness_module._is_parenthesized_outline_label("II")
    assert completeness_module._parenthesized_legal_outline_categories("II") == (
        "upper-alpha",
    )
    assert completeness_module._parenthesized_legal_outline_categories("cc") == (
        "lower-alpha",
        "roman",
    )


# The U.S. Code levels from paragraph or subsection down, with three labels
# each, so no list reaches (H), (h) or (hh), where a label has two readings.
_PARAGRAPH_ROOT_LEVELS = (
    "123",
    "ABC",
    ("i", "ii", "iii"),
    ("I", "II", "III"),
    ("aa", "bb", "cc"),
    ("AA", "BB", "CC"),
)
_SUBSECTION_ROOT_LEVELS = ("abc", *_PARAGRAPH_ROOT_LEVELS)


@st.composite
def _us_code_outlines(draw, *, levels):
    """A U.S. Code outline's text and its paths in document order.

    Every list has two or three members. A parent ends its line with a dash,
    or, below the first level, carries no text and fuses with its first child,
    as in "(II)(aa) Duty."
    """

    expected: list[str] = []
    budget = draw(st.integers(0, 30))

    def render(level: int, path: tuple[str, ...], count: int) -> list[str]:
        nonlocal budget
        lines: list[str] = []
        for index in range(count):
            label = levels[level][index]
            node_path = (*path, label.lower())
            expected.append("/".join(node_path))
            children = 0
            if level + 1 < len(levels) and budget >= 2:
                children = draw(st.sampled_from((0, 0, 2, 3)))
                budget -= children
            child_lines = render(level + 1, node_path, children)
            if children and level > 0 and draw(st.booleans()):
                lines.append(f"({label}){child_lines[0]}")
                lines.extend(child_lines[1:])
            else:
                lines.append(f"({label}) Duty {'—' if children else '.'}")
                lines.extend(child_lines)
        return lines

    return "\n".join(render(0, (), draw(st.integers(2, 3)))) + "\n", expected


@settings(max_examples=300, deadline=None)
@given(_us_code_outlines(levels=_PARAGRAPH_ROOT_LEVELS))
def test_generated_paragraph_outlines_parse_back_to_their_paths(outline):
    text, expected = outline

    assert _paths(text) == expected


@settings(max_examples=300, deadline=None)
@given(_us_code_outlines(levels=_SUBSECTION_ROOT_LEVELS))
def test_generated_subsection_outlines_parse_back_to_their_paths(outline):
    text, expected = outline

    assert _paths(text) == expected
