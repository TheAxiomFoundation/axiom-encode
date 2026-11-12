"""Roman items under paragraph (h) parse as roman items, not as paragraph (i).

``(i)`` is both the letter after ``(h)`` and the first roman numeral. A roman
``(i)`` has ``(ii)`` as its next sibling and a lettered ``(i)`` has ``(j)``, so
the recognizer reads ``(i)`` as roman when the next marker at its level is
``(ii)``. A child ``(A)`` can follow either reading, so after ``(A)`` the
recognizer looks past the children of ``(i)`` for that marker, counting every
label of each marker. The reference filter can drop a marker between ``(i)``
and ``(ii)``, so ``(ii)`` counts only when the unfiltered markers also put it
next. A list can open mid-sentence, as in "may not (i) call; (ii) write", and
no marker regex takes its ``(i)``, so ``(ii)`` also has to stand next when the
recognizer counts the ``(i)`` and ``(ii)`` left in running text.
"""

import heapq

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


def _path_at(text: str, start: int, citation: str = _CITATION) -> tuple[str, ...]:
    """Return the path of the branch whose marker starts at ``start``."""

    (path,) = [
        branch.path
        for branch in recognize_source_structure(text, corpus_citation_path=citation)
        if branch.kind in {"paragraph", "letter"} and branch.start == start
    ]
    return path


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


@pytest.mark.parametrize(
    "items_after_i",
    (
        "(ii) Duty A ii.\n\n(B) Duty B, unless the rate in (ii) applies.",
        "(ii) Duty A ii, unless the rate in (ii) applies.\n\n(B) Duty B.",
    ),
)
def test_a_cited_ii_leaves_a_fused_paragraph_i_a_paragraph(items_after_i: str):
    """A cited (ii) after the items under (A) leaves paragraph (i) a paragraph.

    Invariant: the marker checks pair the (i) fused into "(i)(A)(i)" with the
    (ii) after it and find (j) next, so paragraph (i) reads as a paragraph.
    The recognizer counts the (i) and (ii) in running text only when both
    marker checks put (ii) next, so a cited "(ii)" in running text can't turn
    this paragraph into a roman item.
    """

    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefgh")
        + f"\n\n(i)(A)(i) Duty A i.\n\n{items_after_i}\n\n"
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
        # The inline (i) pairs with the inline (ii), so the later "(ii)" in
        # running text stands unpaired among the (i) and (ii) tokens. Only the
        # unfiltered markers, which put the inline (i) next, keep this
        # paragraph a paragraph.
        (
            _lettered(*"abcdefgh")
            + "\n\n(i) Definitions. In this section: (i) X means x; (ii) Y means y, "
            "as adjusted under (ii) of the table.\n\n(j) Duty j.",
            _CITATION,
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
    ("paragraph_i", "paths_i", "citation"),
    (
        (
            "(i) A collector may not (i) call; (ii) write.\n\n(j) Duty j.",
            [("i",), ("j",)],
            _CITATION,
        ),
        (
            "(i) A collector may not (i) call; (ii) write; or (iii) visit.\n\n"
            "(j) Duty j.",
            [("i",), ("j",)],
            _CITATION,
        ),
        # No (j) follows, so the scan runs to the end of the outline.
        ("(i) A collector may not (i) call; (ii) write.", [("i",)], _CITATION),
        # A drafter skipped the numbered level, and the list sits in (A).
        (
            "(i) Duty i:\n\n(A) A collector may not (i) call; (ii) write.\n\n"
            "(B) Duty B.\n\n(j) Duty j.",
            [("i",), ("i", "a"), ("i", "b"), ("j",)],
            _CITATION,
        ),
        (
            "(i) A person who (i) resides here; (ii) works here is covered.\n\n"
            "(j) Duty j.",
            [("i",), ("j",)],
            _WYOMING_CITATION,
        ),
        # A second list follows "In this section:". The reference filter drops
        # its (i), but the count reads the unfiltered markers, so that (i)
        # still pairs with its (ii).
        (
            "(i) A collector may not (i) call; (ii) write. In this section: "
            "(i) X means x; (ii) Y means y.\n\n(j) Duty j.",
            [("i",), ("j",)],
            _CITATION,
        ),
    ),
)
def test_paragraph_i_whose_text_opens_a_list_mid_sentence_stays_a_paragraph(
    paragraph_i: str, paths_i: list[tuple[str, ...]], citation: str
):
    """A list that opens mid-sentence leaves paragraph (i) a paragraph.

    Invariant: when a lettered paragraph (i), or its child (A), holds a list
    whose (i) follows a word, that (i) pairs with the list's (ii), as does the
    (i) of a later list that the reference filter drops. So (j), or the end of
    the outline, stands next at the level of paragraph (i), and the outline
    parses as main parses it, in the CFR and Wyoming orders.
    """

    text = _lettered(*"abcdefgh") + f"\n\n{paragraph_i}"

    assert _paths(text, citation) == [*_A_TO_G, ("h",), *paths_i]


@pytest.mark.parametrize(
    ("text", "citation"),
    (
        # Oregon drafts (1)(a)(A)(i), and the list sits in (A).
        (
            "(1) Duty one:\n\n"
            + _lettered(*"abcdefgh")
            + "\n\n(i) Duty i:\n\n(A) A person who (i) resides here; (ii) works "
            "here.\n\n(B) Duty B.\n\n(j) Duty j.\n\n(2) Duty two.",
            _OREGON_CITATION,
        ),
        # New Zealand drafts (1)(a)(i)(A), and the list sits in paragraph (i).
        (
            "(1) Duty one:\n\n"
            + "\n\n".join(f"({letter}) Duty {letter};" for letter in "abcdefgh")
            + "\n\n(i) a person who (i) resides here; (ii) works here; and\n\n"
            "(j) Duty j.\n\n(2) Duty two.",
            _NZ_CITATION,
        ),
    ),
)
def test_paragraph_i_whose_text_opens_a_list_stays_a_paragraph_under_1(
    text: str, citation: str
):
    """In the Oregon and New Zealand orders, the paragraph stays (1)(i).

    Invariant: a lettered paragraph (i) under subsection (1) whose text, or
    whose child's text, opens a list mid-sentence reads as paragraph (1)(i),
    and nothing nests under (1)(h). Main also gives the list's inline (ii) a
    branch of its own, which this rule leaves alone, so the test checks only
    (h) and (i).
    """

    paths = _paths(text, citation)

    assert _path_at(text, text.index("\n\n(i) ") + 2, citation) == ("1", "i")
    assert [path for path in paths if path[:2] == ("1", "h")] == [("1", "h")]


def test_fused_paragraph_i_whose_child_opens_a_list_stays_a_paragraph():
    """A fused (i)(A)(i) whose child (B) opens a list stays paragraph (1)(i).

    Invariant: the count of the (i) and (ii) in running text starts with the
    labels fused after (A), so the fused (i) pairs with the (ii) under (A),
    the list's (i) pairs with the list's (ii), and (j) stands next at the
    level of paragraph (i). Main also gives the list's inline (ii) a branch
    of its own, which this rule leaves alone, so the test checks only (h) and
    the branches under (i).
    """

    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefgh")
        + "\n\n(i)(A)(i) Duty A i.\n\n(ii) Duty A ii.\n\n(B) A person who (i) "
        "resides here; (ii) works here.\n\n(j) Duty j.\n\n(2) Duty two."
    )
    paths = _paths(text, _OREGON_CITATION)

    assert [path for path in paths if path[:2] == ("1", "h")] == [("1", "h")]
    assert [path for path in paths if path[:2] == ("1", "i")] == [
        ("1", "i"),
        ("1", "i", "a"),
        ("1", "i", "a", "i"),
        ("1", "i", "a", "ii"),
        ("1", "i", "b"),
    ]


def test_fused_paragraph_i_whose_first_item_holds_a_list_stays_a_paragraph():
    """Keep the fused child when pairing a list in its own running text.

    This models an Oregon paragraph whose first (A)(i) item lists prohibited
    collection methods. Invariant: the fused child (i) and the list's (i)
    each pair with an (ii), leaving (j) next at paragraph (i)'s level. The
    inline (ii) also gets a duplicate branch, so check the paragraph's path.
    """

    text = (
        "(1) Duty one:\n\n"
        + _lettered(*"abcdefgh")
        + "\n\n(i)(A)(i) A collector may not (i) call; (ii) write.\n\n"
        "(ii) Duty A ii.\n\n(j) Duty j.\n\n(2) Duty two."
    )

    paths = _paths(text, _OREGON_CITATION)

    assert ("1", "i") in paths
    assert [path for path in paths if path[:2] == ("1", "h")] == [("1", "h")]


@pytest.mark.parametrize(
    "item_i",
    (
        "(i) A collector may call if (i) asked; (ii) allowed; and",
        "(i) A collector may call if (i) asked, (ii) allowed, and",
    ),
)
def test_roman_item_whose_text_holds_a_list_stays_roman(item_i: str):
    """A roman (i) under (h)(2) stays roman when its own text holds a list.

    Invariant: the list's (i) pairs with the list's (ii), so the (ii) after
    the item still stands next at its level. The inline (ii) after "; " also
    gets an h/2/ii branch of its own, a duplicate that this rule doesn't
    address, so the test checks only the item and its sibling.
    """

    text = _lettered(*"abcdefg") + (
        f"\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two:\n\n{item_i}\n\n(ii) Duty ii."
    )

    assert _path_at(text, text.index(item_i)) == ("h", "2", "i")
    assert _path_at(text, text.index("(ii) Duty ii.")) == ("h", "2", "ii")


@pytest.mark.parametrize(
    "cited",
    (
        # Reference context leads into the cited (i).
        "as limited by clause (i) of this paragraph",
        # A ")", a digit, a letter or a "]" comes right before it.
        "except as paragraph (h)(1)(i) provides",
        "except as 12 CFR 1006.6(i) provides",
        "as column B(i) of the table shows",
        "as note [1](i) explains",
        # A "(" comes right after it, directly or after a space.
        "except as (i)(A) of this paragraph provides",
        "except as (i) (A) of this paragraph provides",
        # Only (i) and (ii) count in running text, so a cited (j) doesn't.
        "except as (j) provides",
    ),
)
def test_roman_item_whose_text_cites_an_item_stays_roman(cited: str):
    """A roman (i) under (h)(2) stays roman when its text cites an item.

    Invariant: a cited (i) opens no list, so it doesn't pair with the (ii)
    after the item, and a cited (j) doesn't stand next at the level of (i).
    The recognizer counts no (i) in running text that reference context
    leads into, that a letter, digit, ")" or "]" comes right before, or that
    a "(" comes right after, and it counts no (j) in running text at all.
    """

    text = _lettered(*"abcdefg") + (
        "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two:\n\n"
        f"(i) Duty i, {cited}.\n\n(ii) Duty ii."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("h", "2", "i"),
        ("h", "2", "ii"),
    ]


def test_a_bare_i_in_a_roman_items_text_gives_the_paragraph_reading():
    """Pin a known limitation: a bare "(i)" in a roman item's own text.

    A roman (i) under (h)(2) whose text mentions "(i)" mid-sentence, outside
    reference context and before the item's (ii), reads as paragraph (i), as
    main reads it. The recognizer takes that mention for the first item of a
    list that opens mid-sentence, pairs it with the (ii), and so withholds the
    roman reading. Invariant pinned: such an item keeps main's reading. A
    rule that tells a mention from a list would change this test.
    """

    text = _lettered(*"abcdefg") + (
        "\n\n(h) Duty h—(1) Duty one.\n\n(2) Duty two:\n\n"
        "(i) The amounts in (i) are final.\n\n(ii) Duty ii."
    )

    assert _paths(text) == [
        *_A_TO_G,
        ("h",),
        ("h", "1"),
        ("h", "2"),
        ("i",),
        ("ii",),
    ]


@pytest.mark.parametrize(
    "paragraph_i",
    (
        # Reference context leads into the list's (i).
        "(i) In this section (i) X means x; (ii) Y means y.",
        # A letter comes right before the list's (i).
        "(i) A collector may not(i) call; (ii) write.",
        # The text also cites a bare "(ii)".
        "(i) A collector may not (i) call; (ii) write, as (ii) of the table says.",
    ),
)
def test_a_list_the_count_misses_gives_the_roman_reading(paragraph_i: str):
    """Pin a known limitation: lists in paragraph (i)'s text that the count misses.

    The recognizer counts an (i) in running text only where it can't be a
    citation, but it counts every (ii) that no reference context leads into.
    When it skips a list's (i), or the text also cites a bare "(ii)", one
    (ii) stands unpaired after paragraph (i), so (i) reads as a roman item
    under (h), where main reads paragraph (i). Invariant pinned: these texts
    read as roman items under (h). A rule that sees these lists would change
    this test.
    """

    text = _lettered(*"abcdefgh") + f"\n\n{paragraph_i}\n\n(j) Duty j."

    assert _paths(text) == [*_A_TO_G, ("h",), ("h", "i"), ("h", "ii"), ("j",)]


_RWANDA_CITATION = (
    "rw/statute/law-2023-048/revenue-of-decentralised-entities/document-1"
)
# The corpus row for Rwanda's law 2023-048, from (h) through the list items. The
# row merges the gazette's columns, so the English definition of «tax stamp»
# runs into a French list.
_RWANDA_TAX_STAMP = (
    "(h) «taxable product » means product manufactured in Rwanda or imported into "
    "Rwanda subject to excise duty;\n\n"
    "(i) «tax stamp » means a sign affixed on a taxable product subject to excise "
    "duty (i) une personne physique et son conjoint et leurs ascendants en ligne "
    "directe ou leurs descendants en ligne directe, et leurs parents en ligne "
    "collatérale jusqu’au 3ème degré au moins;\n\n"
    "(ii) l’une des personnes qui participe directement ou indirectement dans la "
    "gestion, contrôle ou capital d’une autre personne;\n\n"
    "(iii) une tierce personne qui participe directement ou indirectement dans la "
    "gestion, contrôle ou capital ou les deux d’une autre personne ;\n\n"
    "(iv) de telles personnes visées aux sous-points (i), (ii) et (iii) qui "
    "participent directement ou indirectement dans la gestion, contrôle ou capital "
    "d’une entreprise ;"
)


def test_rwanda_tax_stamp_definition_stays_paragraph_i():
    """The «tax stamp» definition in the Rwanda corpus text stays paragraph (i).

    Invariant: a lettered paragraph (i) whose text runs into a list that opens
    mid-sentence reads as paragraph (i), and nothing nests under (h). The
    list's items (ii) to (iv) read as paragraphs of their own, as on main, so
    the test checks only (h) and (i).
    """

    text = _lettered(*"abcdefg") + "\n\n" + _RWANDA_TAX_STAMP
    paths = _paths(text, _RWANDA_CITATION)

    assert _path_at(text, text.index("(i) «tax stamp"), _RWANDA_CITATION) == ("i",)
    assert [path for path in paths if path[0] == "h"] == [("h",)]


@pytest.mark.parametrize(
    ("text", "tokens"),
    (
        ("may not (i) call; (ii) write.", ["(i)"]),
        ("may call if (i) asked, (ii) allowed.", ["(i)", "(ii)"]),
        ("as limited by clause (i) of this paragraph.", []),
        ("as paragraph (h)(1)(i) and 12 CFR 1006.6(ii) provide.", []),
        ("as column B(i) and note [1](ii) show.", []),
        ("as (i)(A) and (ii) (B) provide.", []),
        ("(i) Duty i.\n(ii) Duty ii.", []),
        ("Duty: (i) X; (ii) Y.", []),
        ("as (I) and (iii) show.", []),
    ),
)
def test_markers_with_standalone_roman_items(text: str, tokens: list[str]):
    """The token stream adds only the (i) and (ii) that no marker regex takes.

    Invariant: the stream holds every marker, in document order, plus each
    lower-case (i) or (ii) that sits outside every marker, follows no letter,
    digit, ")" or "]", precedes no "(", and has no reference context.
    """

    markers = tuple(
        heapq.merge(
            completeness_module._PARENTHESIZED_OUTLINE_MARKER.finditer(text),
            completeness_module._INLINE_PARENTHESIZED_OUTLINE_MARKER.finditer(text),
            key=lambda match: match.start("marker"),
        )
    )
    merged = completeness_module._markers_with_standalone_roman_items(text, markers)
    starts = [match.start("marker") for match in merged]
    marker_starts = {match.start("marker") for match in markers}

    assert starts == sorted(starts)
    assert marker_starts <= set(starts)
    assert [
        match.group("marker")
        for match in merged
        if match.start("marker") not in marker_starts
    ] == tokens


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
_WYOMING_LEVELS = ("lower-alpha", "roman", "upper-alpha", "upper-roman")
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
    if category == "upper-roman":
        return _ROMAN[index].upper()
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


# Lists that open mid-sentence: the (i) follows a word, so it opens no marker,
# and the (ii) follows "; ", so the inline regex takes it.
_MID_SENTENCE_LISTS = (
    " who (i) acts; (ii) waits",
    " who (i) acts; (ii) waits; or (iii) rests",
    " if (i) asked; (ii) allowed, as the rules say",
)


@st.composite
def _outlines_with_a_list_in_paragraph_i(draw, *, levels: tuple[str, ...]):
    """An outline past (i) whose lettered paragraph (i) opens a list mid-sentence.

    Returns the text, the paths of the outline without the list, and where the
    marker of paragraph (i) starts. Only paragraphs use the word "alpha", so
    the list goes into the one line that opens "(i) Duty for alpha".
    """

    text, expected = draw(_outlines_past_i(levels=levels))
    paragraph_i = "(i) Duty for alpha"
    assert text.count(paragraph_i) == 1
    sublist = draw(st.sampled_from(_MID_SENTENCE_LISTS))
    return (
        text.replace(paragraph_i, paragraph_i + sublist),
        expected,
        text.index(paragraph_i),
    )


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


@settings(max_examples=300, deadline=None, derandomize=True)
@given(_outlines_with_a_list_in_paragraph_i(levels=_CFR_LEVELS))
def test_generated_cfr_outlines_with_a_list_in_paragraph_i_parse_back(outline):
    """A list that opens mid-sentence in paragraph (i) changes no CFR path.

    Invariant: a CFR-order outline past (i), whose lettered paragraph (i)
    opens a list mid-sentence, parses back to the paths of the outline
    without the list. So paragraph (i) stays a paragraph, and the list gets
    no branch.
    """

    text, expected, _paragraph_i_start = outline

    assert _paths(text) == expected


@settings(max_examples=300, deadline=None, derandomize=True)
@given(_outlines_with_a_list_in_paragraph_i(levels=_WYOMING_LEVELS))
def test_generated_wyoming_paragraph_i_with_a_list_stays_a_paragraph(outline):
    """A list that opens mid-sentence leaves Wyoming's paragraph (i) a paragraph.

    Invariant: in the Wyoming order, (a)(i)(A)(I), a lettered paragraph (i)
    that opens a list mid-sentence reads as paragraph (i). Main and this rule
    both misparse some Wyoming outlines below that level, such as an (I) under
    (A), so the test checks only the path of paragraph (i).
    """

    text, _expected, paragraph_i_start = outline

    assert _path_at(text, paragraph_i_start, _WYOMING_CITATION) == ("i",)
