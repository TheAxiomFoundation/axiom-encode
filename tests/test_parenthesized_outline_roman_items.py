"""Roman items under paragraph (h) parse as roman items, not as paragraph (i).

``(i)`` is both the letter after ``(h)`` and the first roman numeral. A roman
``(i)`` has ``(ii)`` as its next sibling and a lettered ``(i)`` has ``(j)``, so
the recognizer reads ``(i)`` as roman when the next marker at its level is
``(ii)``. A child ``(A)`` can follow either reading, so after ``(A)`` the
recognizer looks past the children of ``(i)`` for that marker, counting every
label of each marker. The reference filter can drop a marker between ``(i)``
and ``(ii)``, so ``(ii)`` counts only when the unfiltered markers also put it
next.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import recognize_source_structure

_CITATION = "us/regulation/12/1006/14"
_OREGON_CITATION = "us-or/statute/316/680"
_NZ_CITATION = "nz/act/2007/97/section/YA1"
_WYOMING_CITATION = "us-wy/statute/39/13/103"

# 12 CFR 1006.14(h), as the corpus serves it to the encoder.
_PROHIBITED_MEDIA = (
    "(h) Prohibited communication media—(1) In general. In connection with the "
    "collection of any debt, a debt collector must not communicate or attempt to "
    "communicate with a person through a medium of communication if the person "
    "has requested that the debt collector not use that medium to communicate "
    "with the person.\n\n"
    "(2) Exceptions. Notwithstanding the prohibition in paragraph (h)(1) of this "
    "section:\n\n"
    "(i) If a person opts out of receiving electronic communications from a debt "
    "collector, a debt collector may send an electronic confirmation of the "
    "person's request to opt out, provided that the electronic confirmation "
    "contains no information other than a statement confirming the person's "
    "request and that the debt collector will honor it;\n\n"
    "(ii) If a person initiates contact with a debt collector using a medium of "
    "communication that the person previously requested the debt collector not "
    "use, the debt collector may respond once through the same medium of "
    "communication used by the person; or\n\n"
    "(iii) If otherwise required by applicable law, a debt collector may "
    "communicate or attempt to communicate with a person in connection with the "
    "collection of any debt through a medium of communication that the person has "
    "requested the debt collector not use to communicate with the person."
)
_A_TO_G = [(letter,) for letter in "abcdefg"]


def _paths(text: str, citation: str = _CITATION) -> list[tuple[str, ...]]:
    return [
        branch.path
        for branch in recognize_source_structure(text, corpus_citation_path=citation)
        if branch.kind in {"paragraph", "letter"}
    ]


def _lettered(*letters: str) -> str:
    return "\n\n".join(f"({letter}) Duty {letter}." for letter in letters)


def _under(root: str, paths: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return [(root, *path) for path in paths]


def test_roman_items_under_paragraph_h_nest_under_h_2():
    text = _lettered(*"abcdefg") + "\n\n" + _PROHIBITED_MEDIA

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("h", "2", "i"),
        ("h", "2", "ii"),
        ("h", "2", "iii"),
    ]


def test_roman_item_with_children_under_paragraph_h_nests_under_h_2():
    text = _lettered(*"abcdefg") + (
        "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two:\n\n(i) Duty i:\n\n"
        "(A) Duty A;\n\n(B) Duty B.\n\n(ii) Duty ii."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("h", "2", "i"),
        ("h", "2", "i", "a"),
        ("h", "2", "i", "b"),
        ("h", "2", "ii"),
    ]


def test_roman_items_take_the_level_below_their_parent():
    text = _lettered(*"abcdefg") + (
        "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two:\n\n(i) Duty i:\n\n"
        "(A) Duty A:\n\n(1) Duty one:\n\n(i) Duty i:\n\n(A) Duty A.\n\n(B) Duty B.\n\n"
        "(ii) Duty ii.\n\n(2) Duty two.\n\n(B) Duty B.\n\n(ii) Duty ii."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("h", "2", "i"),
        ("h", "2", "i", "a"),
        ("h", "2", "i", "a", "1"),
        ("h", "2", "i", "a", "1", "i"),
        ("h", "2", "i", "a", "1", "i", "a"),
        ("h", "2", "i", "a", "1", "i", "b"),
        ("h", "2", "i", "a", "1", "ii"),
        ("h", "2", "i", "a", "2"),
        ("h", "2", "i", "b"),
        ("h", "2", "ii"),
    ]


@pytest.mark.parametrize(
    ("item_i", "paths_i"),
    (
        (
            "(i)(A) Duty A.\n\n(B) Duty B.",
            [("h", "2", "i"), ("h", "2", "i", "a"), ("h", "2", "i", "b")],
        ),
        (
            "(i)(A)(1) Duty one.\n\n(2) Duty two.\n\n(B) Duty B.",
            [
                ("h", "2", "i"),
                ("h", "2", "i", "a"),
                ("h", "2", "i", "a", "1"),
                ("h", "2", "i", "a", "2"),
                ("h", "2", "i", "b"),
            ],
        ),
    ),
)
def test_roman_item_fused_with_its_first_child_nests_under_h_2(
    item_i: str, paths_i: list[tuple[str, ...]]
):
    text = _lettered(*"abcdefg") + (
        f"\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two:\n\n{item_i}\n\n(ii) Duty ii."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        *paths_i,
        ("h", "2", "ii"),
    ]


def test_a_nested_roman_v_does_not_end_the_search_for_ii():
    # (v) reads as a letter or a roman numeral. Here it is the fifth roman item
    # under (h)(2)(i)(A)(1), and the search runs past it to (ii).
    text = _lettered(*"abcdefg") + (
        "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two:\n\n(i) Duty i:\n\n"
        "(A) Duty A:\n\n(1) Duty one:\n\n"
        + "\n\n".join(
            f"({item}) Item {item}." for item in ("i", "ii", "iii", "iv", "v")
        )
        + "\n\n(2) Duty two.\n\n(B) Duty B.\n\n(ii) Duty ii."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("h", "2", "i"),
        ("h", "2", "i", "a"),
        ("h", "2", "i", "a", "1"),
        *[("h", "2", "i", "a", "1", item) for item in ("i", "ii", "iii", "iv", "v")],
        ("h", "2", "i", "a", "2"),
        ("h", "2", "i", "b"),
        ("h", "2", "ii"),
    ]


def test_paragraph_i_after_h_stays_a_paragraph():
    followed_by_j = _lettered(*"abcdefg") + (
        "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two.\n\n(i) Duty i.\n\n(j) Duty j."
    )
    with_children = _lettered(*"abcdefg") + (
        "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two.\n\n"
        "(i) Duty i—(1) Duty one.\n\n(2) Duty two."
    )

    assert _paths(followed_by_j) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("i",),
        ("j",),
    ]
    assert _paths(with_children) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("i",),
        ("i", "1"),
        ("i", "2"),
    ]


def test_paragraph_i_whose_first_child_is_a_capital_stays_a_paragraph():
    # A drafter skipped the numbered level, so (A) follows paragraph (i).
    text = _lettered(*"abcdefgh") + (
        "\n\n(i) Duty i:\n\n(A) Duty A.\n\n(B) Duty B.\n\n(j) Duty j."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("i",),
        ("i", "a"),
        ("i", "b"),
        ("j",),
    ]


@pytest.mark.parametrize(
    ("paragraph_h", "paths_h"),
    (
        ("(h) Duty h.", [("h",)]),
        (
            "(h) Duty h:\n\n(A) Duty A.\n\n(B) Duty B.",
            [("h",), ("h", "a"), ("h", "b")],
        ),
    ),
)
def test_oregon_paragraph_i_with_subparagraphs_stays_a_paragraph(
    paragraph_h: str, paths_h: list[tuple[str, ...]]
):
    # Oregon drafts (1)(a)(A)(i), so a lettered paragraph's children are (A).
    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefg")
        + f"\n\n{paragraph_h}\n\n(i) Duty i:\n\n(A) Duty A.\n\n(B) Duty B.\n\n"
        "(j) Duty j.\n\n(2) Duty two."
    )

    assert _paths(text, _OREGON_CITATION) == [
        ("1",),
        *_under("1", _A_TO_G),
        *_under("1", paths_h),
        ("1", "i"),
        ("1", "i", "a"),
        ("1", "i", "b"),
        ("1", "j"),
        ("2",),
    ]


def test_oregon_last_paragraph_i_stays_a_paragraph():
    # No (j) follows (i). The next lettered paragraph, (2)(a), ends the search.
    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefgh")
        + "\n\n(i) Duty i:\n\n(A) Duty A.\n\n(B) Duty B.\n\n"
        "(2) Duty two:\n\n(a) Duty a.\n\n(b) Duty b."
    )

    assert _paths(text, _OREGON_CITATION) == [
        ("1",),
        *_under("1", [*_A_TO_G, ("h",)]),
        ("1", "i"),
        ("1", "i", "a"),
        ("1", "i", "b"),
        ("2",),
        ("2", "a"),
        ("2", "b"),
    ]


@pytest.mark.parametrize(
    ("subparagraph_i", "paths_i"),
    (
        ("(i) Duty i.", [("1", "h", "i")]),
        (
            "(i) Duty i:\n\n(A) Duty A.\n\n(B) Duty B.",
            [("1", "h", "i"), ("1", "h", "i", "a"), ("1", "h", "i", "b")],
        ),
    ),
)
def test_new_zealand_subparagraphs_under_h_nest_under_h(
    subparagraph_i: str, paths_i: list[tuple[str, ...]]
):
    # New Zealand drafts (1)(a)(i)(A), so paragraph (h)'s children are roman.
    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefg")
        + f"\n\n(h) Duty h:\n\n{subparagraph_i}\n\n(ii) Duty ii.\n\n(2) Duty two."
    )

    assert _paths(text, _NZ_CITATION) == [
        ("1",),
        *_under("1", _A_TO_G),
        ("1", "h"),
        *paths_i,
        ("1", "h", "ii"),
        ("2",),
    ]


def test_oregon_paragraph_i_with_roman_items_under_A_stays_a_paragraph():
    # The (ii) under (A) pairs with the (i) under (A), so (j) stands next.
    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefgh")
        + "\n\n(i) Duty i:\n\n(A) Duty A:\n\n(i) Duty A i.\n\n(ii) Duty A ii.\n\n"
        "(B) Duty B.\n\n(j) Duty j.\n\n(2) Duty two."
    )

    assert _paths(text, _OREGON_CITATION) == [
        ("1",),
        *_under("1", [*_A_TO_G, ("h",)]),
        ("1", "i"),
        ("1", "i", "a"),
        ("1", "i", "a", "i"),
        ("1", "i", "a", "ii"),
        ("1", "i", "b"),
        ("1", "j"),
        ("2",),
    ]


@pytest.mark.parametrize(
    "paragraph_i",
    (
        "(i) Duty i:\n\n(A)(i) Duty A i.",
        "(i) Duty i:\n\n(A) (i) Duty A i.",
        "(i)(A)(i) Duty A i.",
    ),
)
def test_oregon_paragraph_i_stays_a_paragraph_when_A_fuses_with_its_first_item(
    paragraph_i: str,
):
    # The (i) inside the fused marker pairs with the (ii) after it.
    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefgh")
        + f"\n\n{paragraph_i}\n\n(ii) Duty A ii.\n\n(B) Duty B.\n\n"
        "(j) Duty j.\n\n(2) Duty two."
    )

    assert _paths(text, _OREGON_CITATION) == [
        ("1",),
        *_under("1", [*_A_TO_G, ("h",)]),
        ("1", "i"),
        ("1", "i", "a"),
        ("1", "i", "a", "i"),
        ("1", "i", "a", "ii"),
        ("1", "i", "b"),
        ("1", "j"),
        ("2",),
    ]


def test_paragraph_i_whose_capital_fuses_with_a_numbered_item_stays_a_paragraph():
    # A drafter skipped the numbered level, and (A) fuses with its own (1)(i).
    text = _lettered(*"abcdefgh") + (
        "\n\n(i) Duty i:\n\n(A)(1)(i) Duty.\n\n(ii) Duty.\n\n(2) Duty.\n\n"
        "(B) Duty B.\n\n(j) Duty j."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("i",),
        ("i", "a"),
        ("i", "a", "1"),
        ("i", "a", "1", "i"),
        ("i", "a", "1", "ii"),
        ("i", "a", "2"),
        ("i", "b"),
        ("j",),
    ]


_INLINE_DEFINITIONS = (
    "\n\n(i) Definitions. In this section: (i) X means x; (ii) Y means y."
    "\n\n(j) Duty j."
)


@pytest.mark.parametrize(
    ("text", "citation"),
    (
        (_lettered(*"abcdefgh") + _INLINE_DEFINITIONS, _CITATION),
        (
            _lettered(*"abcdefg")
            + "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two."
            + _INLINE_DEFINITIONS,
            _CITATION,
        ),
        # Wyoming drafts (a)(i)(A)(I), so a subsection's paragraphs are roman.
        (
            _lettered(*"abcdefgh")
            + '\n\n(i) As used in this section: (i) "X" means x; (ii) "Y" means y.'
            + "\n\n(j) Duty j.",
            _WYOMING_CITATION,
        ),
    ),
)
def test_paragraph_i_stays_a_paragraph_when_the_filter_drops_its_inline_i(
    text: str, citation: str
):
    # The reference filter drops the inline (i) after "this section:" and
    # keeps the (ii), so only the unfiltered markers show (i) before (ii).
    paths = _paths(text, citation)

    assert [path for path in paths if len(path) == 1] == [
        (letter,) for letter in "abcdefghij"
    ]
    assert not [path for path in paths if path[0] == "h" and path[-1] in _ROMAN]


@pytest.mark.parametrize(
    ("after_i", "expected"),
    (
        ("(A) a.\n(B) b.\n(ii) c.", "ii"),
        ("(A) a.\n(B) b.\n(j) c.", "j"),
        # A nested (i) pairs with the (ii) after it.
        ("(A) a:\n(i) p.\n(ii) q.\n(B) b.\n(j) c.", "j"),
        ("(A) a:\n(i) p.\n(ii) q.\n(B) b.\n(ii) c.", "ii"),
        # A nested (i) with no (ii) leaves the reading to the other rules.
        ("(A) a:\n(i) p.\n(B) b.\n(j) c.", None),
        # A roman-looking letter such as (v) doesn't end the search.
        ("(A) a:\n(1) p:\n(v) q.\n(B) b.\n(ii) c.", "ii"),
        # Any other lower-case letter does.
        ("(A) a.\n(B) b.\n(k) c.\n(ii) d.", None),
        ("(A) a.\n(B) b.", None),
        # Every label of a marker counts, so a fused (i) pairs as well.
        ("(A)(i) p.\n(ii) q.\n(B) b.\n(j) c.", "j"),
        ("(A)(i) p.\n(ii) q.\n(B) b.\n(ii) c.", "ii"),
        ("(A) a:\n(1)(i) p.\n(ii) q.\n(B) b.\n(j) c.", "j"),
    ),
)
def test_next_label_beside_i(after_i: str, expected: str | None):
    matches = tuple(
        completeness_module._PARENTHESIZED_OUTLINE_MARKER.finditer(
            f"(i) Duty i:\n{after_i}"
        )
    )

    assert completeness_module._next_label_beside_i(matches, 1) == expected


@pytest.mark.parametrize(
    ("leading", "expected"),
    ((("i",), "j"), ((), "ii"), (("1", "i"), "j")),
)
def test_next_label_beside_i_reads_the_labels_after_A_first(
    leading: tuple[str, ...], expected: str
):
    # In "(i)(A)(i)", the labels after (A) open the children of (A).
    matches = tuple(
        completeness_module._PARENTHESIZED_OUTLINE_MARKER.finditer(
            "(i)(A)(i) p.\n(ii) q.\n(B) b.\n(j) c."
        )
    )

    assert completeness_module._next_label_beside_i(matches, 1, leading) == expected


@pytest.mark.parametrize(
    ("label", "opens"),
    (
        ("ii", True),
        ("II", False),
        ("A", False),
        ("a", False),
        ("I", False),
        ("1", False),
        ("j", False),
        ("iii", False),
        (None, False),
    ),
)
def test_only_ii_reads_i_as_roman(label: str | None, opens: bool):
    assert completeness_module._outline_label_opens_roman_items(label) is opens


# Generated outlines take the first four levels of a drafting order, such as
# the 12 CFR order (a), (1), (i), (A). Below that the CFR order repeats (1) and
# (i), and flat text can't tell whether a (3) after "(B)(2)" continues the
# inner list or the outer one, so the generator stops at the fourth level.
# Lists stay short and paragraphs stop at (n), so no letter reads as a later
# roman numeral such as (v) or (x), and only (i) after (h) has two readings.
_CFR_LEVELS = ("lower-alpha", "numeric", "roman", "upper-alpha")
_NZ_LEVELS = ("numeric", "lower-alpha", "roman", "upper-alpha")
_OREGON_LEVELS = ("numeric", "lower-alpha", "upper-alpha", "roman")
_ATTACHMENTS = ("colon", "dash", "fused")
_ROMAN = ("i", "ii", "iii")
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_WORDS = ("alpha", "beta", "gamma", "delta")


def _label(category: str, index: int) -> str:
    if category == "lower-alpha":
        return _ALPHABET[index]
    if category == "upper-alpha":
        return _ALPHABET[index].upper()
    if category == "numeric":
        return str(index + 1)
    return _ROMAN[index]


def _render(nodes: list, levels: tuple[str, ...]) -> tuple[str, list[tuple[str, ...]]]:
    """Render ``(index, attachment, children)`` nodes and list their paths.

    A parent ends its line with a colon, attaches its first child after a
    dash, as in "(h) Prohibited communication media—(1) In general.", or,
    below the first level, carries no text and fuses with its first child,
    as in "(A)(i) Duty for delta."
    """

    expected: list[tuple[str, ...]] = []

    def render(nodes: list, level: int, path: tuple[str, ...]) -> list[str]:
        lines: list[str] = []
        for index, attachment, children in nodes:
            label = _label(levels[level], index)
            node_path = (*path, label.lower())
            expected.append(node_path)
            head = f"({label}) Duty for {_WORDS[level]}"
            child_lines = render(children, level + 1, node_path)
            if children and attachment == "fused" and level > 0:
                lines.append(f"({label}){child_lines[0]}")
                lines.extend(child_lines[1:])
            elif children and attachment == "dash":
                lines.append(f"{head}—{child_lines[0]}")
                lines.extend(child_lines[1:])
            else:
                lines.append(f"{head}{':' if children else '.'}")
                lines.extend(child_lines)
        return lines

    return "\n\n".join(render(nodes, 0, ())), expected


@st.composite
def _outlines(draw, *, max_paragraphs: int):
    """A CFR-order outline's text and its paths in document order.

    Every list has at least two members, as drafted law does.
    """

    budget = draw(st.integers(0, 40))

    attachments = st.sampled_from(_ATTACHMENTS)

    def build(level: int) -> list[tuple[int, str, list]]:
        nonlocal budget
        if level >= len(_CFR_LEVELS) or budget < 2:
            return []
        count = draw(st.sampled_from((0, 0, 0, 2, 2, 3)))
        if count:
            budget -= count
        return [(index, draw(attachments), build(level + 1)) for index in range(count)]

    paragraphs = draw(st.integers(2, max_paragraphs))
    tree = [(index, draw(attachments), build(1)) for index in range(paragraphs)]
    return _render(tree, _CFR_LEVELS)


@st.composite
def _outlines_past_i(draw, *, levels: tuple[str, ...]):
    """An outline whose lettered list runs to (i) or past it.

    Paragraphs (h) and (i) always have children, and their first child often
    has children of its own, so most outlines hold an (i) of each reading
    after (h). In an order with a numeric first level, the lettered list sits
    under "(1)", and a "(2)" follows it.
    """

    lettered = levels.index("lower-alpha")
    budget = draw(st.integers(0, 24))
    attachments = st.sampled_from(_ATTACHMENTS)

    def build(level: int, forced_depth: int = 0) -> list[tuple[int, str, list]]:
        nonlocal budget
        if level >= len(levels) or (budget < 2 and not forced_depth):
            return []
        count = draw(st.sampled_from((2, 2, 3) if forced_depth else (0, 0, 0, 2, 2, 3)))
        budget -= count
        return [
            (
                index,
                draw(attachments),
                build(level + 1, forced_depth - 1 if index == 0 else 0),
            )
            for index in range(count)
        ]

    paragraphs = [
        (
            index,
            draw(attachments),
            build(lettered + 1, draw(st.integers(1, 3)) if index in (7, 8) else 0),
        )
        for index in range(draw(st.integers(9, 12)))
    ]
    tree = [(0, "colon", paragraphs), (1, "colon", [])] if lettered else paragraphs
    return _render(tree, levels)


@settings(max_examples=400, deadline=None)
@given(_outlines(max_paragraphs=14))
def test_generated_outlines_parse_back_to_their_paths(outline):
    text, expected = outline

    assert _paths(text) == expected


@settings(max_examples=300, deadline=None)
@given(_outlines_past_i(levels=_CFR_LEVELS))
def test_generated_cfr_outlines_past_i_parse_back_to_their_paths(outline):
    text, expected = outline

    assert _paths(text) == expected


@settings(max_examples=300, deadline=None)
@given(_outlines_past_i(levels=_NZ_LEVELS))
def test_generated_new_zealand_outlines_past_i_parse_back_to_their_paths(outline):
    text, expected = outline

    assert _paths(text, _NZ_CITATION) == expected


@settings(max_examples=300, deadline=None)
@given(_outlines_past_i(levels=_OREGON_LEVELS))
def test_generated_oregon_outlines_past_i_parse_back_to_their_paths(outline):
    text, expected = outline

    assert _paths(text, _OREGON_CITATION) == expected
