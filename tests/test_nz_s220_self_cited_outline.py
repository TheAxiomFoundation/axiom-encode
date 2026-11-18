"""A complete section's own subsection citation authenticates its bare outline.

Excerpt: Social Security Act 2018 (2018/32), section 220, from the pinned
`nz-rulespec-2026-07-25` release (see `tests/fixtures/nz_bare_labels/README.md`).
Source: New Zealand Legislation (legislation.govt.nz), Crown copyright, CC BY 4.0

Every adversarial input from the #1992 reviews must keep base's obligations
under a complete-section path, the only scope in which the new evidence counts.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode import corpus_resolver
from axiom_encode.bare_structure import (
    bare_outline,
    is_complete_section_path,
    mask_bare_labels,
)
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp
from tests.scaling import assert_linear_growth, lowest_work
from tests.test_bare_structure_nz_harness import (
    HEADED_RATES_TABLE,
    MARKER_ONLY_RATES_TABLE,
    MARKER_ONLY_REPLACEMENT_TABLE,
    NESTED_COUNT_SOURCES,
    RATES_TABLE,
    REPLACEMENT_TABLE,
    S73,
    S220,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nz_bare_labels"
SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
S220_PATH = "nz/statute/act/public/2018/0032/section/220"
JURISDICTIONS = ("nz", "uk", "us", "de", "fr")
SECTION_PATHS = (
    S220_PATH,
    *(f"{j}/statute/example/section/15" for j in JURISDICTIONS),
    *(f"{j}/statute/example/section/220" for j in JURISDICTIONS),
)
SELF_CITATION = "under subsection (2)"


def numbers(source: str, path: str) -> list[float]:
    cleaned = sc.authoritative_numeric_recall_text(source, corpus_citation_path=path)
    return vp.extract_numeric_occurrences_from_text(cleaned)


def unchanged_from_base(source: str, path: str) -> None:
    assert bare_outline(source, corpus_citation_path=path) == bare_outline(source)
    assert mask_bare_labels(source, corpus_citation_path=path) == mask_bare_labels(
        source
    )


def test_fixture_is_the_pinned_s220_excerpt():
    assert (FIXTURES / "s220.txt").read_text() == S220 + "\n"


# Admitted examples. --------------------------------------------------------


@pytest.mark.parametrize("ending", ("\n", "\r\n", "\u2028"))
def test_complete_s220_owns_its_bare_subsections(ending):
    source = S220.replace("\n", ending)
    outline = bare_outline(source, corpus_citation_path=S220_PATH)
    assert outline is not None
    assert [label.path for label in outline] == [
        ("1",),
        ("2",),
        ("2", "a"),
        ("2", "b"),
        ("3",),
    ]
    exception = outline[3]
    assert "section 72" in source[exception.marker_end : exception.end]
    assert "Sections 219" not in source[exception.start : exception.end]
    # Labels 1 and 2 leave on the citation of subsection (2). Nothing proves
    # root 3, so its label stays a source obligation, as on main.
    assert [label.proven for label in outline] == [True, True, True, True, False]
    assert numbers(source, S220_PATH) == [1, 28, 1, 3]
    branches = sc.recognize_source_structure(source, corpus_citation_path=S220_PATH)
    assert [branch.path for branch in branches] == [label.path for label in outline]


def test_s220_without_a_section_scope_keeps_base_obligations():
    for path in ("", "nz/statute/act/public/2018/0032", "us/statute/26/21"):
        unchanged_from_base(S220, path)
        assert bare_outline(S220, corpus_citation_path=path) is None
        assert numbers(S220, path) == [1, 1, 2, 28, 1, 3]


def test_corpus_resolver_slices_a_self_cited_s220_subsection():
    sliced = corpus_resolver._slice_parent_body(
        S220, requested_path=f"{S220_PATH}/2/b", resolved_path=S220_PATH
    )
    assert sliced.startswith("MSD is satisfied")
    assert sliced.endswith("under section 72 .")


def test_corpus_resolver_refuses_the_unproven_s220_subsection():
    # Root 3 follows the highest cited subsection, so its identity is unproven.
    with pytest.raises(corpus_resolver.CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            S220, requested_path=f"{S220_PATH}/3", resolved_path=S220_PATH
        )


def test_legal_subject_nested_source_is_admitted_once_it_cites_itself():
    # Paired with the #1992 legal-subject control, which has no citation and
    # stays rejected: here subsection (2) exists, and only bare 2 can be it.
    source = NESTED_COUNT_SOURCES["legal-subject"].replace("Family size.\n", "")
    source = source.replace("$450", f"$450 {SELF_CITATION}")
    outline = bare_outline(source, corpus_citation_path=S220_PATH)
    assert outline is not None
    assert [label.path for label in outline][-1] == ("2",)
    assert 450 in numbers(source, S220_PATH)


# Review inputs against the citation evidence. A numeric-only outline stays
# rejected even when it cites itself: `1 The household is entitled to $450 ...`
# with a later line calling each number a household size cites its own
# subsection (2) and is still a count axis. The nested forms put the same
# sentence inside the citing subsection, where only the context check can
# reject it.

NUMERIC_ONLY = (
    "1\nThe household is entitled to $450.\n2\nThe household is entitled to $700.\n"
)
SELF_CITED_NUMERIC_ONLY = (
    f"1\nThe household is entitled to $450 {SELF_CITATION}.\n"
    "2\nThe household is entitled to $700.\n"
)
NESTED_HEAD = (
    "1\nThe household is entitled to $450 if:\n"
    "a\nthe applicant is resident; and\n"
    "b\nthe applicant has no income."
)
NESTED_TAIL = "\n2\nSections 72 and 73 apply."
REVIEW_CITATIONS = (
    "By s 71 subsection (2), each number above is a household size.",
    "By s. 71 subsection (2) the amounts are set.",
    "Title 7, subsection (2), sets the household sizes.",
    "The rates in paragraph 4, subsection (2), apply by household size.",
    "Under the Code, subsection (2) governs.",
    "Subpart 3 subsection (2) applies.",
    "This is replaced in section 71 by the following\nsubsection (2).",
    "The number above each amount is the household size, as provided in\n"
    "subsection (2) .",
    "Each figure printed above an entitlement is the number of persons, "
    "unless subsection (2) applies.",
    "Division 2, subsection (2), applies.",
    "Item 4 subsection (2) applies.",
    "§ 71 subsection (2) applies.",
)


@pytest.mark.parametrize("citation", REVIEW_CITATIONS)
def test_review_citations_do_not_admit_a_numeric_only_outline(citation):
    unchanged_from_base(NUMERIC_ONLY + citation, S220_PATH)
    unchanged_from_base(SELF_CITED_NUMERIC_ONLY + citation, S220_PATH)
    assert numbers(NUMERIC_ONLY + citation, S220_PATH)[:4] == [1, 450, 2, 700]


@pytest.mark.parametrize("citation", REVIEW_CITATIONS)
@pytest.mark.parametrize("joiner", (" ", "\n"))
def test_review_citations_do_not_admit_a_nested_outline(citation, joiner):
    source = NESTED_HEAD + joiner + citation + NESTED_TAIL
    unchanged_from_base(source, S220_PATH)
    assert bare_outline(source, corpus_citation_path=S220_PATH) is None


# Third-review inputs: a citation proves its own root, so a count caption or
# count anaphor beside an uncited legal-unit root keeps that root's number.
COUNT_HEAD = (
    "1\nA payment is made under subsection (2).\n"
    "2\nThe payment is $40 a week for each child if—\n"
    "a\nthe recipient is a sole parent; and\n"
)
COUNT_KEY_INPUTS = {
    "caption-before-root": (
        COUNT_HEAD + "b\nthe recipient has children. Family size.\n"
        "3\nSections 72 and 73 apply.",
        [1, 2, 40, 3],
    ),
    "caption-in-follower": (
        COUNT_HEAD + "b\nthe recipient has children.\n"
        "3\nSections 72 and 73 apply. Family size.\n"
        "4\nSections 74 and 75 apply.",
        [1, 2, 40, 3, 4],
    ),
    "this-many": (
        COUNT_HEAD + "b\nthe recipient has children.\n"
        "3\nSection 74 applies if there are this many or more children.",
        [1, 2, 40, 3],
    ),
    "that-many": (
        COUNT_HEAD + "b\nthe recipient has children.\n"
        "3\nSections 72 and 73 apply where the recipient has that many children "
        "or more.",
        [1, 2, 40, 3],
    ),
    "up-to-that-number": (
        COUNT_HEAD + "b\nthe recipient has children.\n"
        "3\nSections 72 and 73 apply to each of those children, up to that number.",
        [1, 2, 40, 3],
    ),
}


@pytest.mark.parametrize("path", SECTION_PATHS)
@pytest.mark.parametrize("case", COUNT_KEY_INPUTS, ids=COUNT_KEY_INPUTS)
def test_count_words_beside_an_uncited_legal_root_keep_its_number(
    base, receipts, path, case
):
    source, expected = COUNT_KEY_INPUTS[case]
    only_proven_labels_leave(base, receipts, source, path)
    if path.startswith("nz/"):
        # Labels 1 and 2 leave on the citation; every later root stays.
        assert numbers(source, path) == [n for n in expected if n not in (1, 2)]


# Fourth-review inputs. Each keeps main's numbers through structure, not a
# noun list: every text line must end in a clause once a legal-unit root is
# admitted, and a legal-unit follower must not refer back to a quantity.
REVIEW_HEAD = (
    "1\nA payment is made under subsection (2).\n"
    "2\nThe payment is $40 a week if—\n"
    "a\nthe recipient is a sole parent; and\n"
    "b\nthe recipient is resident."
)
LEGAL_ROOT = "\n3\nSections 72 and 73 apply."
REVIEW_CAPTIONS = (
    # Dropped the 3 at the reviewed head.
    "Adults.",
    "Occupants.",
    "Bedrooms.",
    "Weeks of absence.",
    "Days.",
    "Beneficiaries.",
    "Pupils.",
    "Dwellings.",
    "Units.",
    "Partners.",
    "Grandchildren.",
    "Kids.",
    "Tenants.",
    "Rooms.",
    "Instalments.",
    "Hours a week.",
    "Tier.",
    "Scale.",
    "Band.",
    "Level.",
    "Step.",
    "Headcount.",
    "Siblings.",
    # Kept main's numbers at the reviewed head.
    "Persons in the dwelling.",
    "Person count.",
    "Years of service.",
    "Year.",
    "Number.",
    "Quantity.",
    "Total.",
    "Rate band.",
    "Dependants.",
    "Family size.",
    "Household.",
    # Relative clauses are still captions.
    "Adults who are resident.",
    "Persons that are resident.",
)
REVIEW_ANAPHORS = (
    # Dropped the 3 at the reviewed head.
    "Sections 72 and 73 apply to absences of that length in weeks.",
    "Sections 72 and 73 apply for that period of weeks.",
    "Sections 72 and 73 apply to dwellings with more occupants than that.",
    # Kept main's numbers at the reviewed head.
    "Sections 72 and 73 apply for that many weeks.",
    "Sections 72 and 73 apply to as many adults.",
    "Sections 72 and 73 apply to households of that size.",
    "Sections 72 and 73 apply to the corresponding number of adults.",
    "Sections 72 and 73 apply that many times.",
    "Sections 72 and 73 apply to that number of adults.",
    "Sections 72 and 73 apply to children of the following sizes.",
)
REVIEW_LAYOUTS = {
    "line-0-caption": "Family size.\n" + REVIEW_HEAD + LEGAL_ROOT,
    "caption-in-previous-follower": REVIEW_HEAD.replace(
        "subsection (2).", "subsection (2). Family size."
    )
    + LEGAL_ROOT,
    "caption-on-a-later-line": REVIEW_HEAD + LEGAL_ROOT + "\nFamily size.",
}
NZ_SECTION = S220_PATH


HOSTILE_PATHS = (
    NZ_SECTION,
    "us/statute/26/section/1",
    "uk/statute/example/section/15",
)


def only_proven_labels_leave(base, receipts, source, path):
    """Compared with main, only labels the citation proves leave the recall.

    The citation proves the cited roots and the roots before them. Every
    other number main demands, including each later root's label, stays.
    """
    if base.bare_outline(source) is not None:
        # Base admits it: the outline must be base's, with nothing new proven.
        assert label_rows(bare_outline(source, corpus_citation_path=path)) == (
            label_rows(base.bare_outline(source))
        )
        return None
    outline = bare_outline(source, corpus_citation_path=path)
    with receipts.base_masking(base):
        before = Counter(numbers(source, path))
    after = Counter(numbers(source, path))
    assert not after - before
    proven = Counter(
        float(label.token)
        for label in outline or ()
        if len(label.path) == 1 and label.proven and label.token.isdigit()
    )
    assert before - after == proven
    for label in outline or ():
        if len(label.path) == 1 and not label.proven:
            assert after[float(label.token)] == before[float(label.token)]
    return outline


def keeps_unproven_labels(base, receipts, source, kept=(3,)):
    for path in HOSTILE_PATHS:
        only_proven_labels_leave(base, receipts, source, path)
    # Under the NZ path, every listed later number keeps main's count.
    with receipts.base_masking(base):
        before = Counter(numbers(source, NZ_SECTION))
    after = Counter(numbers(source, NZ_SECTION))
    assert all(after[number] == before[number] for number in kept)


def test_the_review_head_is_admitted_without_a_caption_or_anaphor():
    # Paired admitted example: the controls below differ from it by one line.
    source = REVIEW_HEAD + LEGAL_ROOT
    assert bare_outline(source) is None
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is not None
    # Labels 1 and 2 leave on the citation of subsection (2); 3 stays.
    assert numbers(source, NZ_SECTION) == [40, 3]


@pytest.mark.parametrize("caption", REVIEW_CAPTIONS)
def test_a_caption_before_a_legal_root_keeps_main(base, receipts, caption):
    keeps_unproven_labels(base, receipts, REVIEW_HEAD + " " + caption + LEGAL_ROOT)


@pytest.mark.parametrize("follower", REVIEW_ANAPHORS)
def test_an_anaphoric_legal_follower_keeps_main(base, receipts, follower):
    keeps_unproven_labels(base, receipts, REVIEW_HEAD + "\n3\n" + follower)


@pytest.mark.parametrize("layout", REVIEW_LAYOUTS, ids=REVIEW_LAYOUTS)
def test_captions_elsewhere_keep_main(base, receipts, layout):
    keeps_unproven_labels(base, receipts, REVIEW_LAYOUTS[layout])


@pytest.mark.parametrize("caption", ("Family size.", "Adults.", "Tenants."))
def test_a_caption_before_a_prose_root_keeps_main_once_a_legal_root_is_admitted(
    base, receipts, caption
):
    # The new route: main rejects this body only because of root 4, so a
    # caption before prose root 3 must still keep its number here.
    keeps_unproven_labels(
        base,
        receipts,
        REVIEW_HEAD
        + " "
        + caption
        + "\n3\nThe payment is $700.\n4\nSections 72 and 73 apply.",
        kept=(3, 700, 4),
    )


# Fifth-review inputs: clause-shaped captions and back-references outside any
# list, and an anaphor in a prose root. Under the cited-root rule they need no
# rule of their own: root 3 follows the highest cited subsection (2), so its
# label stays whatever the caption or follower says.
CLAUSE_CAPTIONS = (
    "Adults are counted here.",
    "The key is family size.",
    "Adults must be counted.",
    "Children may be counted.",
    "Dependants are counted.",
    "Weeks of absence are counted here.",
    "Occupants are counted by head.",
    "Tiers apply by headcount.",
    "Adults in the household are counted.",
    "Adults is the heading.",
    "Adults, as at 1 April, are counted.",
    "Bands are numbered from 1.",
    # Kept main's numbers at the reviewed head.
    "Each entry is a number of adults.",
    "Rates vary with household size.",
    "Family size determines the rate.",
    "Adults (the rate is fixed).",
    "The left-hand figure is the household size.",
    "Adults—the count is below.",
    "The rate depends on the band.",
    "The band is the number of adults.",
    "Applicable band.",
    "Band applicable.",
    "Adults counted.",
    "Adults per dwelling.",
    "Number of adults, then the rule.",
    "Adults: the rule is in each line",
    "Adults has its count.",
    "The rate is fixed—Adults.",
    "Is the count.",
)
BACK_REFERENCES = tuple(
    f"Sections 72 and 73 apply {tail}"
    for tail in (
        "to an equal number of adults.",
        *(
            f"to households of the {word} size."
            for word in ("stated", "indicated", "given", "like", "equal")
        ),
        "to the indicated number of weeks.",
        "to the matching number of adults.",
        "to so many adults.",
        "to as large a household.",
        "to households no larger.",
        "to fewer adults.",
        "to households thus sized.",
        "to households sized accordingly.",
        "to the adults so numbered.",
        "to adults numbering thus.",
        "for the period stated.",
        "per adult, up to the figure shown.",
        "to a household of the size at left.",
        "to that.",
        "to it.",
        "to them all.",
        "likewise.",
        "to more adults.",
        "to each adult.",
        "to the adults counted.",
        # Kept main's numbers at the reviewed head.
        "to households that large.",
        "to households this big.",
        "to that many.",
    )
)


@pytest.mark.parametrize("caption", CLAUSE_CAPTIONS)
def test_a_clause_shaped_caption_keeps_the_later_root(base, receipts, caption):
    keeps_unproven_labels(base, receipts, REVIEW_HEAD + " " + caption + LEGAL_ROOT)


@pytest.mark.parametrize("follower", BACK_REFERENCES)
def test_a_back_reference_keeps_the_later_root(base, receipts, follower):
    keeps_unproven_labels(base, receipts, REVIEW_HEAD + "\n3\n" + follower)


def test_an_anaphor_in_a_prose_root_keeps_its_label(base, receipts):
    # The regression from `1e94dec`: root 3 is a family size.
    keeps_unproven_labels(
        base,
        receipts,
        REVIEW_HEAD
        + "\n3\nThe payment is $700 for a family of this size."
        + "\n4\nSections 72 and 73 apply.",
        kept=(3, 700, 4),
    )


def every_hostile_input():
    """Every hostile outline input from the four review rounds of this PR."""
    yield from (NUMERIC_ONLY + c for c in REVIEW_CITATIONS)
    yield from (SELF_CITED_NUMERIC_ONLY + c for c in REVIEW_CITATIONS)
    yield from (
        NESTED_HEAD + joiner + c + NESTED_TAIL
        for c in REVIEW_CITATIONS
        for joiner in (" ", "\n")
    )
    yield from (source for source, _expected in COUNT_KEY_INPUTS.values())
    yield from (REVIEW_HEAD + " " + c + LEGAL_ROOT for c in REVIEW_CAPTIONS)
    yield from (REVIEW_HEAD + " " + c + LEGAL_ROOT for c in CLAUSE_CAPTIONS)
    yield from (REVIEW_HEAD + "\n3\n" + f for f in REVIEW_ANAPHORS)
    yield from (REVIEW_HEAD + "\n3\n" + f for f in ANAPHOR_FOLLOWERS)
    yield from (REVIEW_HEAD + "\n3\n" + f for f in BACK_REFERENCES)
    yield from REVIEW_LAYOUTS.values()
    yield from (
        REVIEW_HEAD + f" {c}\n3\nThe payment is $700.\n4\nSections 72 and 73 apply."
        for c in ("Family size.", "Adults.", "Tenants.")
    )
    yield (
        REVIEW_HEAD + "\n3\nThe payment is $700 for a family of this size."
        "\n4\nSections 72 and 73 apply."
    )
    yield (
        "1\nThe household is entitled to $450 unless subsection (2) applies.\n"
        "2\nThe household is entitled to $700 if it has a disabled member.\n"
        "Each figure printed above an entitlement is the number of persons in "
        "the household."
    )
    yield (
        "1\nThe household is entitled to $450 under subsection (2) if—\n"
        "a\nit is resident; and\nb\nit has a dependent child.\n"
        "2\nSections 5 and 6 do not apply.\n3\nSections 7 and 8 apply.\n"
    )


def test_no_hostile_input_drops_an_obligation_main_keeps(base, receipts):
    sources = list(dict.fromkeys(every_hostile_input()))
    assert len(sources) > 150
    admitted = 0
    for source in sources:
        for path in HOSTILE_PATHS:
            if only_proven_labels_leave(base, receipts, source, path):
                admitted += 1
    # Many are admitted, so the invariant is exercised, not skipped.
    assert admitted > 100


# Accepted behaviour, pinned so any change to it is deliberate. A caption at
# the end of 1(b) sits before the cited subsection (3), so roots 1 and 2 are
# proven by order: the section has a subsection 3, it is the bare `3`, and the
# walk from line 0 has no gap, duplicate or other marker, so the bare `2` is
# the only unit that can be subsection 2. A repeal before a live subsection
# would leave a gap instead, which the walk rejects (see the gap tests below).
CAPTIONED_ROOT = (
    "1\nThe payment is $40 a week if—\n"
    "a\nthe recipient is a sole parent; and\n"
    "b\nthe recipient is resident. Family size.\n"
    "2\nSections 72 and 73 apply.\n"
    "3\nThe payment is not payable under subsection (3) to a person in prison."
)


def test_a_caption_before_a_cited_root_is_accepted_on_the_order_proof(base, receipts):
    assert base.bare_outline(CAPTIONED_ROOT) is None
    with receipts.base_masking(base):
        assert numbers(CAPTIONED_ROOT, NZ_SECTION) == [1, 40, 2, 3]
    outline = bare_outline(CAPTIONED_ROOT, corpus_citation_path=NZ_SECTION)
    assert [(label.path, label.proven) for label in outline] == [
        (("1",), True),
        (("1", "a"), True),
        (("1", "b"), True),
        (("2",), True),
        (("3",), True),
    ]
    assert numbers(CAPTIONED_ROOT, NZ_SECTION) == [40]


LETTER_SUFFIX = (
    "1\nThe payment is payable under subsection (2).\n"
    "2\nThe payment is payable if—\n"
    "a\nthe person is resident; and\n"
    "b\nthe person has no income.\n"
    "2A\nSections 219 and 436 do not apply to the payment.\n"
    "3\nSections 72 and 73 apply."
)


def test_an_inserted_root_after_the_cited_one_stays_unproven():
    # `2A` follows the cited `(2)` in the order, so a citation of `(2)` proves
    # 1 and 2 only; `2A` and 3 keep their labels and can't be sliced.
    outline = bare_outline(LETTER_SUFFIX, corpus_citation_path=NZ_SECTION)
    assert [(label.token, label.proven) for label in outline] == [
        ("1", True),
        ("2", True),
        ("a", True),
        ("b", True),
        ("2A", False),
        ("3", False),
    ]
    with pytest.raises(corpus_resolver.CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            LETTER_SUFFIX, requested_path=f"{NZ_SECTION}/2A", resolved_path=NZ_SECTION
        )


REPEALED_BODY = (
    "1\nThe payment is payable to the person under subsection (3).\n"
    "2\n{second}\n"
    "3\nThe payment is payable if—\n"
    "a\nthe person is resident; and\n"
    "b\nthe person has no income.\n"
    "4\nSections 72 and 73 apply."
)


def test_the_repealed_body_control_is_admitted():
    source = REPEALED_BODY.format(second="The payment is payable to the person.")
    assert bare_outline(source) is None
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is not None


@pytest.mark.parametrize(
    "second",
    (
        "[Repealed]",
        "Repealed.",
        "[Repealed by section 5 of the Amendment Act 2020]",
        "Repealed by section 5 of the Amendment Act 2020.",
    ),
)
def test_a_rendered_repealed_subsection_keeps_main(base, receipts, second):
    # The NZ corpus renders no tombstone: it drops a repealed subsection (see
    # the gap tests below). These guard other renderings a source could use.
    source = REPEALED_BODY.format(second=second)
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is None
    for path in HOSTILE_PATHS:
        only_proven_labels_leave(base, receipts, source, path)


@pytest.mark.parametrize(
    "second",
    (
        "[Repealed on 1 April 2020 by section 5 of the Amendment Act 2020 "
        "(2020 No 5).]",
        "Repealed, on 1 April 2020, by section 5 of the Amendment Act 2020 "
        "(2020 No 5).",
        "Subsection (2) was repealed on 1 April 2020.",
    ),
)
def test_more_rendered_tombstones_fail_the_walk(base, receipts, second):
    source = REPEALED_BODY.format(second=second)
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is None
    for path in HOSTILE_PATHS:
        only_proven_labels_leave(base, receipts, source, path)


@pytest.mark.parametrize(
    "second",
    (
        "This subsection was repealed on 1 April 2020.",
        "The subsection is repealed.",
    ),
)
def test_a_prose_tombstone_leaves_the_subsection_its_own_label(base, receipts, second):
    # Admitted, and correctly so: the bare `2` really is subsection 2's label.
    source = REPEALED_BODY.format(second=second)
    outline = bare_outline(source, corpus_citation_path=NZ_SECTION)
    assert [(label.token, label.proven) for label in outline] == [
        ("1", True),
        ("2", True),
        ("3", True),
        ("a", True),
        ("b", True),
        ("4", False),
    ]
    with receipts.base_masking(base):
        assert numbers(source, NZ_SECTION) == [1, 2, 3, 4]
    assert numbers(source, NZ_SECTION) == [4]


@pytest.mark.parametrize("before", (None, "3"))
def test_a_composed_body_never_reaches_the_order_proof(before):
    # `corpus_resolver._compose_descendant_text` omits repealed descendants and
    # joins the rest with blank lines. A blank line before a numeric root fails
    # main's walk, so a composed body can't let a count key fill a gap.
    lines = S220.split("\n")
    joined = [
        f"\n{line}"
        if index and line.isdigit() and (before is None or line == before)
        else line
        for index, line in enumerate(lines)
    ]
    source = "\n".join(joined)
    assert source != S220
    assert bare_outline(source, corpus_citation_path=S220_PATH) is None


def test_an_omitted_subsection_leaves_a_gap_the_walk_rejects():
    source = REPEALED_BODY.replace("2\n{second}\n", "")
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is None


# The corpus's real shape for a repeal: the NZ corpus drops a repealed
# subsection's label and text, so the live bare roots skip its number. Built
# from the s220 fixture by renumbering, with the citation following its root.
REPEAL_GAP = (
    S220.replace("\n3\nSections", "\n4\nSections")
    .replace("\n2\nThe winter", "\n3\nThe winter")
    .replace("under subsection (2).", "under subsection (3).")
)
LEADING_REPEAL_GAP = REPEAL_GAP.replace("1\nThe winter", "2\nThe winter", 1)


def test_the_repeal_gap_control_is_the_s220_fixture():
    outline = bare_outline(S220, corpus_citation_path=NZ_SECTION)
    assert [label.path for label in outline] == [
        ("1",),
        ("2",),
        ("2", "a"),
        ("2", "b"),
        ("3",),
    ]


@pytest.mark.parametrize(
    ("source", "roots"),
    ((REPEAL_GAP, ["1", "3", "4"]), (LEADING_REPEAL_GAP, ["2", "3", "4"])),
    ids=("gap-after-1", "leading-gap"),
)
def test_a_repeal_gap_in_the_bare_roots_proves_nothing(source, roots):
    lines = source.split("\n")
    assert [line for line in lines if line.isdigit()] == roots
    assert [line for line in lines if line in ("a", "b")] == ["a", "b"]
    assert "subsection (3)" in source
    for path in (NZ_SECTION, ""):
        assert bare_outline(source, corpus_citation_path=path) is None
        assert mask_bare_labels(source, corpus_citation_path=path) == source
    # Every number main demands stays, including each bare root.
    recalled = numbers(source, NZ_SECTION)
    assert all(float(root) in recalled for root in roots)


# Word-level tests for every surviving word list: each word alone defeats the
# evidence in a body that is otherwise admitted.

CHILD_CITED = (
    "1\nThe payment is payable if—\n"
    "a\nthe person is resident under subsection (2){word}; and\n"
    "b\nthe person has no income.\n"
    "2\nThe payment is payable to the person.\n"
    "3\nSections 219 and 436 do not apply to the payment."
)
FOREIGN_WORDS = (
    "s ss sch section sections act acts code codes title titles statute statutes "
    "law laws ordinance ordinances order orders decree decrees instrument "
    "instruments provision provisions schedule schedules part parts subpart "
    "subparts division divisions subdivision subdivisions chapter chapters "
    "article articles paragraph paragraphs subparagraph subparagraphs clause "
    "clauses subclause subclauses item items rule rules regulation regulations "
    "replace replaces replaced insert inserts inserted repeal repeals repealed "
    "omit omits omitted substitute substitutes substituted amend amends amended "
    "amendment amendments following size sizes figure figures number numbers "
    "numeral numerals count counts persons people"
).split()
FOREIGN_SYMBOLS = ('"', "“", "”", "‘", "’", "«", "»", "§", "¶")


def test_the_child_cited_body_is_admitted():
    source = CHILD_CITED.format(word="")
    assert bare_outline(source) is None
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is not None


@pytest.mark.parametrize("word", FOREIGN_WORDS)
def test_each_foreign_word_defeats_the_citation(word):
    source = CHILD_CITED.format(word=f" {word}")
    unchanged_from_base(source, NZ_SECTION)
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is None


@pytest.mark.parametrize("symbol", FOREIGN_SYMBOLS)
def test_each_foreign_symbol_defeats_the_citation(symbol):
    source = CHILD_CITED.format(word=f" {symbol}x")
    unchanged_from_base(source, NZ_SECTION)
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is None


ANAPHOR_FOLLOWERS = (
    "Sections 72 and 73 apply to that payment.",
    "Sections 72 and 73 apply to this payment.",
    "Sections 72 and 73 apply to those payments.",
    "Sections 72 and 73 apply to these payments.",
    "Sections 72 and 73 apply more than once.",
    *(
        f"Sections 72 and 73 apply as {word} as needed."
        for word in ("many", "much", "few", "little", "long", "often")
    ),
    *(
        f"Sections 72 and 73 apply to the {word} payment."
        for word in (
            "corresponding",
            "respective",
            "same",
            "following",
            "preceding",
            "foregoing",
            "above",
            "aforementioned",
            "said",
        )
    ),
    "Sections 72 and 73 apply to such payments.",
    "Sections 72 and 73 apply to the payment below.",
)


@pytest.mark.parametrize("follower", ANAPHOR_FOLLOWERS)
def test_each_anaphor_defeats_a_legal_root(base, receipts, follower):
    keeps_unproven_labels(base, receipts, REVIEW_HEAD + "\n3\n" + follower)


@pytest.mark.parametrize(
    "line",
    (
        "The household is resident.",
        "The household are resident.",
        "MSD must pay.",
        "MSD shall pay.",
        "MSD may pay.",
        "MSD has paid.",
        "MSD does not pay.",
        "The payment was made.",
        "The payments were made.",
        "MSD cannot pay.",
        "MSD will pay.",
        "Section 5 applies.",
    ),
)
def test_a_clause_before_a_legal_root_is_admitted(line):
    # Paired with the captions: a line ending in a main clause passes.
    source = REVIEW_HEAD + " " + line + LEGAL_ROOT
    assert bare_outline(source, corpus_citation_path=NZ_SECTION) is not None


@pytest.mark.parametrize(
    "source",
    (
        # The `of`/`in` qualifier: this subsection belongs to another unit.
        S220.replace(
            "under subsection (2).", "under subsection (2) of the principal enactment."
        ),
        # A self-citation never lifts the nested-label requirement.
        "1\nThe payment is payable under subsection (2).\n2\nSections 72 and 73 apply.",
        # Value words in the citing unit.
        S220.replace(
            "unless it is so payable under subsection (2).",
            "unless it is so payable to those persons under subsection (2).",
        ),
        # A legal-unit subject must be followed directly by its predicate.
        S220.replace(
            "Sections 219 and 436 do not apply",
            "Sections 219 and 436 also do not apply",
        ),
    ),
    ids=("qualified", "numeric-only", "value-words", "no-predicate"),
)
def test_each_evidence_condition_is_required(source):
    unchanged_from_base(source, S220_PATH)
    assert bare_outline(source, corpus_citation_path=S220_PATH) is None


def test_self_cited_numeric_only_outline_with_a_count_statement_stays_rejected():
    source = (
        "1\nThe household is entitled to $450 unless subsection (2) applies.\n"
        "2\nThe household is entitled to $700 if it has a disabled member.\n"
        "Each figure printed above an entitlement is the number of persons in "
        "the household."
    )
    unchanged_from_base(source, S220_PATH)
    assert numbers(source, S220_PATH) == [1, 450, 2, 700]


@pytest.mark.parametrize(
    "marker", ("payment(2) is payable.", "x( 2 ) The payment.", "（2） The payment.")
)
def test_every_other_spelling_of_the_cited_marker_defeats_the_citation(marker):
    source = S220.replace("lasts in total)", "lasts in total) " + marker)
    unchanged_from_base(source, S220_PATH)


def many_citations(count, *, clean):
    if clean:
        return S220.replace(
            "under subsection (2).",
            "under subsection (2)." + " It is payable under subsection (2)." * count,
        )
    return S220 + "\nThis applies under subsection (1). " * count


def many_roots(count):
    lines = [f"1\nThe payment is payable under subsection ({count}).", "2", CHAPEAU]
    lines.append(CHILDREN)
    lines.extend(
        f"{number}\nSections 219 and 436 do not apply to the payment."
        for number in range(3, count + 1)
    )
    return "\n".join(lines)


def outline_work(source, *, admitted):
    def check(outline):
        assert (outline is not None) == admitted

    return lowest_work(
        lambda: bare_outline(source, corpus_citation_path=S220_PATH), check
    )


@pytest.mark.parametrize("clean", (False, True), ids=("foreign-unit", "clean-unit"))
def test_many_citations_stay_linear(clean):
    # A per-citation rescan of the unit grew quadratically: about 54x at 8x
    # input, failing in seconds at these sizes; the head grows about 4x.
    small = outline_work(many_citations(25, clean=clean), admitted=clean)
    large = outline_work(many_citations(200, clean=clean), admitted=clean)
    assert_linear_growth(small, large, input_ratio=8)


def test_cost_stays_linear_in_the_number_of_roots():
    # The former per-root marker scan grew about 64x at 8x the roots.
    small = outline_work(many_roots(500), admitted=True)
    large = outline_work(many_roots(4000), admitted=True)
    assert_linear_growth(small, large, input_ratio=8)


# Negative controls: the #1992 review inputs, now under section scope. ---------


@pytest.mark.parametrize("path", SECTION_PATHS)
def test_detached_s73_keeps_numeric_obligations_under_section_scope(path):
    # Blocks bare-number s73: it cites none of its own subsections, and the
    # `1 The household is entitled to $450. 2 ...` count axis has its shape.
    unchanged_from_base(S73, path)
    assert re.findall(r"\d+", mask_bare_labels(S73, corpus_citation_path=path)) == [
        "1",
        "2",
        "3",
        "4",
    ]


@pytest.mark.parametrize("path", SECTION_PATHS)
@pytest.mark.parametrize("heading", ("Family size.", "Household size.", "Age.", ""))
def test_detached_count_axes_stay_rejected_under_section_scope(path, heading):
    source = (
        (heading + "\n" if heading else "")
        + "1\nThe household is entitled to $450.\n"
        + "2\nThe household is entitled to $700."
    )
    unchanged_from_base(source, path)
    assert numbers(source, path) == [1, 450, 2, 700]


@pytest.mark.parametrize("path", SECTION_PATHS)
@pytest.mark.parametrize("heading", ("Family size.", "Age."))
@pytest.mark.parametrize("case", NESTED_COUNT_SOURCES, ids=NESTED_COUNT_SOURCES)
def test_nested_count_axes_stay_rejected_under_section_scope(path, heading, case):
    source = NESTED_COUNT_SOURCES[case].replace("Family size.", heading, 1)
    unchanged_from_base(source, path)
    expected = [1, 450, 2] if case == "legal-subject" else [1, 450, 2, 700]
    assert numbers(source, path) == expected


@pytest.mark.parametrize("path", SECTION_PATHS)
@pytest.mark.parametrize("heading", ("Family size.", "Age.", "Winter energy payment."))
@pytest.mark.parametrize("case", NESTED_COUNT_SOURCES, ids=NESTED_COUNT_SOURCES)
def test_a_heading_before_the_outline_defeats_a_self_citation(path, heading, case):
    # A caption may introduce a count axis, so the walk must open the body.
    source = NESTED_COUNT_SOURCES[case].replace("Family size.", heading, 1)
    source = source.replace("$450", f"$450 {SELF_CITATION}")
    unchanged_from_base(source, path)
    assert 1 in numbers(source, path)


@pytest.mark.parametrize("path", SECTION_PATHS)
@pytest.mark.parametrize(
    "source",
    (
        RATES_TABLE,
        HEADED_RATES_TABLE,
        REPLACEMENT_TABLE,
        MARKER_ONLY_RATES_TABLE,
        MARKER_ONLY_REPLACEMENT_TABLE,
        "1 | The amount is $450 under subsection (2).\n2 | The amount is $700.\n",
        "Number of children | Eligibility\n"
        "1 | The child is eligible under subsection (2).\n"
        "2 | The children are eligible.",
    ),
)
def test_tables_stay_rejected_under_section_scope(path, source):
    unchanged_from_base(source, path)


@pytest.mark.parametrize("path", SECTION_PATHS)
@pytest.mark.parametrize(
    "quantity", ("rate prescribed", "amount specified", "percentage approved")
)
def test_prescribed_quantity_complements_stay_rejected(path, quantity):
    unchanged_from_base(S73.replace("rate prescribed", quantity), path)


# Negative controls for the new evidence itself. -------------------------------


@pytest.mark.parametrize(
    "path",
    (
        "",
        f"{S220_PATH}/2",
        f"{S220_PATH}/2/a",
        "nz/statute/act/public/2018/0032/schedule/4/part/7/clause/4",
        "nz/statute/act/public/2018/0032",
        "us/statute/26/21",
        "section/220",
        f"{S220_PATH}(1)",
        f"{S220_PATH}#rate",
        f"{S220_PATH}.1",
        "us-ri/guidance/revenue/section/12",
        "nz/statute/act/public/2018/0032/schedule/4/section/1",
        "NZ/statute/act/public/2018/0032/section/220",
        "nz/statute/act/public/2018/0032/schedule-4/section/1",
        "nz/statute/act/public/2018/0032/schedules/4/section/1",
        "nz/statute/regulation/2018/0202/sch/4/section/1",
        "nz/statute/act/public/2018/0032/first-schedule/section/1",
        "nz/statute/act/public/2018/0032/appendix/4/section/1",
        "nz/statute/act/public/2018/0032/annex/4/section/1",
        "nz/statute/act/public/2018/0032/table/4/section/1",
        "nz/statute/act/public/2018/0032/repealed/section/220",
        f"{S220_PATH}-repealed",
        f"{S220_PATH}-2",
        f"{S220_PATH}/section/2",
        "nz/statute/act/public/2018/0032/section/-",
        "nz/statute/act/public/2018/0032/form/2/section/1",
        "nz/statute/act/public/2018/0032/forms/section/1",
    ),
)
def test_only_a_complete_section_path_counts(path):
    unchanged_from_base(S220, path)
    assert bare_outline(S220, corpus_citation_path=path) is None


@pytest.mark.parametrize(
    "citation",
    (
        "under subsection (2) of section 71",
        "under subsection (2) in Schedule 4",
        "under subsection (2) as inserted by section 15",
        "under subsection (2) of the Act",
        "under “subsection (2)”",
        "under subsection (2), as replaced",
        "under subsection (2) of Part 2",
        "under subsection (5)",
        "under subsections (2) and (5)",
        "under this section",
        "under subsection 2",
        "under section 220(2)",
    ),
)
def test_citations_that_do_not_name_a_bare_root_prove_nothing(citation):
    source = S220.replace("under subsection (2)", citation)
    assert source != S220
    unchanged_from_base(source, S220_PATH)
    assert bare_outline(source, corpus_citation_path=S220_PATH) is None


@pytest.mark.parametrize(
    "marker",
    (" (2) The payment is payable.", " (3) The payment is payable."),
)
def test_another_parenthesized_marker_could_be_the_cited_subsection(marker):
    # `(2)` beside the citation could be subsection 2 itself; `(3)` could be a
    # third subsection that the bare `3` is not.
    source = S220.replace("lasts in total)", "lasts in total)" + marker)
    unchanged_from_base(source, S220_PATH)


def test_foreign_reference_anywhere_on_the_citing_line_proves_nothing():
    source = S220.replace(
        "The winter energy payment is not payable",
        "Despite section 64, the winter energy payment is not payable",
    )
    unchanged_from_base(source, S220_PATH)


def test_every_other_walk_check_still_applies_under_section_scope():
    # Main rejects an unresolved quantity before a numeric root; the citation
    # proves the outline's kind, not that `3` does not complete `maximum`.
    reduced = S220.replace(
        " if—\na\nthe payment would be payable to the beneficiary were it not for "
        "those days of those absences; and\nb\nMSD is satisfied that those days of "
        "those absences do not affect the beneficiary’s eligibility for the payment "
        "under section 72 .",
        ".",
    )
    assert reduced != S220
    unchanged_from_base(reduced, S220_PATH)


# Invariants. ------------------------------------------------------------------
#
# The bodies come from an outline grammar, so base admits many of them and the
# self-citation admits others; `test_the_grammar_reaches_every_verdict` fails
# if either kind stops being generated. Each property compares the current
# module with base's, loaded from git.

FOLLOWERS = (
    "The payment is payable to the person.",
    "The payment is payable under subsection ({cite}).",
    "Sections 219 and 436 do not apply to the payment.",
    "Sections 72 and 73 apply to absences of that length.",
    "The household is entitled to $450.",
    "Section 74 applies under subsection ({cite}) of section 71.",
)
CHAPEAU = "The payment is payable if—"
CHILDREN = "a\nthe person is resident; and\nb\nthe person has no income."
GRAMMAR_PATHS = (
    "",
    S220_PATH,
    f"{S220_PATH}/2",
    "us/statute/26/21",
    "uk/statute/example/section/9",
)


GRAMMAR_CAPTIONS = ("Family size.", "Tenants.", "Adults who are resident.")


def grammar_body(heading, roots):
    lines = [heading] if heading else []
    for number, (follower, children, caption, cite) in enumerate(roots, start=1):
        if caption and lines:
            lines[-1] = lines[-1] + " " + GRAMMAR_CAPTIONS[number % 3]
        lines.append(str(number))
        if children:
            lines.extend((CHAPEAU, CHILDREN))
        else:
            lines.append(FOLLOWERS[follower].format(cite=cite))
    return "\n".join(lines)


_ROOT = st.tuples(
    st.integers(0, len(FOLLOWERS) - 1),
    st.booleans(),
    st.booleans(),
    st.integers(1, 4),
)
_GRAMMAR = st.builds(
    grammar_body,
    st.sampled_from(("", "Winter energy payment.")),
    st.lists(_ROOT, min_size=2, max_size=4),
)


def label_rows(outline):
    if outline is None:
        return None
    return [
        (label.token, label.path, label.start, label.marker_end, label.end)
        for label in outline
    ]


@pytest.fixture(scope="module")
def receipts():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import nz_harness_defect_receipts as module
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    return module


@pytest.fixture(scope="module")
def base(receipts):
    return receipts.load_base()


def check_against_base(base, source, path):
    before = label_rows(base.bare_outline(source))
    current = label_rows(bare_outline(source, corpus_citation_path=path))
    # Without a section path every input keeps base's verdict.
    assert label_rows(bare_outline(source)) == before
    if not is_complete_section_path(path):
        assert current == before
    # Whatever base admits is unchanged by any path.
    if before is not None:
        assert current == before
        return "base"
    if current is None:
        return "rejected"
    # A new admission needs scope, a walk from line 0 and a cited bare root.
    assert is_complete_section_path(path)
    assert current[0][2] == 0
    outline = bare_outline(source, corpus_citation_path=path)
    roots = [label for label in outline if len(label.path) == 1]
    # Own citations only: `subsection (N) of section 71` names another unit.
    own = {int(m) for m in re.findall(r"subsection \((\d+)\)(?! of)", source)}
    cited = [int(r.token) for r in roots if int(r.token) in own]
    assert cited
    # Only the cited roots and the ones before them are proven; every later
    # root, and everything under it, keeps its marker in the masked text.
    masked = mask_bare_labels(source, corpus_citation_path=path)
    for label in outline:
        root = int(label.path[0])
        assert label.proven == (root <= max(cited))
        marker = source[label.start : label.marker_end]
        assert masked[label.start : label.marker_end] == (
            " " * len(marker) if label.proven else marker
        )
    uncited = re.sub(r"subsections? \(", "item (", source)
    assert bare_outline(uncited, corpus_citation_path=path) is None
    return "new"


@settings(max_examples=500, deadline=None)
@given(source=_GRAMMAR, path=st.sampled_from(GRAMMAR_PATHS))
@example(
    source="a\nThe household is entitled to $450.\nb\nThe household is entitled to $450.",
    path="",
)
def test_outlines_match_base_except_on_self_cited_evidence(base, source, path):
    check_against_base(base, source, path)


def grammar_cases():
    roots = [
        (follower, children, caption, 2)
        for follower in range(len(FOLLOWERS))
        for children in (False, True)
        for caption in (False, True)
    ]
    for first in roots:
        for second in roots:
            for third in roots:
                source = grammar_body("", (first, second, third))
                for path in ("", S220_PATH, "us/statute/26/21"):
                    yield source, path


def test_the_grammar_reaches_every_verdict(base):
    # Exhaustive over three-root bodies citing subsection (2): the properties
    # above run on each, and base admissions, new admissions and rejections
    # must all occur, so the properties cannot pass vacuously.
    seen = {}
    for source, path in grammar_cases():
        verdict = check_against_base(base, source, path)
        seen[verdict] = seen.get(verdict, 0) + 1
    assert seen.get("base", 0) >= 50
    assert seen.get("new", 0) >= 50
    assert seen.get("rejected", 0) >= 50


@settings(max_examples=200, deadline=None)
@given(
    amounts=st.lists(st.integers(min_value=4, max_value=99999), min_size=3, max_size=3)
)
def test_admitted_mask_hides_only_labels_and_keeps_every_amount(amounts):
    source = (
        f"1\nThe payment is ${amounts[0]} {SELF_CITATION}.\n"
        f"2\nThe payment is ${amounts[1]} if—\n"
        "a\nthe applicant is resident; and\n"
        "b\nthe applicant has no income.\n"
        f"3\nSections 219 and 436 do not apply to the ${amounts[2]} payment."
    )
    outline = bare_outline(source, corpus_citation_path=S220_PATH)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    assert len(masked) == len(source)
    spans = [(label.start, label.marker_end) for label in outline if label.proven]
    for index, (original, kept) in enumerate(zip(source, masked, strict=True)):
        inside = any(start <= index < end for start, end in spans)
        assert kept == (" " if inside else original)
    # Root 3 follows the cited subsection (2), so its label stays.
    assert numbers(source, S220_PATH) == [amounts[0], amounts[1], 3, amounts[2]]


def _drop_last_label_of_base_outlines(real):
    def planted(text, *, corpus_citation_path=""):
        outline = real(text, corpus_citation_path=corpus_citation_path)
        if is_complete_section_path(corpus_citation_path) and outline:
            if real(text) is not None:
                return outline[:-1]
        return outline

    return planted


def _reject_some_lengths_without_scope(real):
    def planted(text, *, corpus_citation_path=""):
        if not is_complete_section_path(corpus_citation_path) and len(text) % 7 == 0:
            return None
        return real(text, corpus_citation_path=corpus_citation_path)

    return planted


@pytest.mark.parametrize(
    "plant", (_drop_last_label_of_base_outlines, _reject_some_lengths_without_scope)
)
def test_the_properties_catch_planted_bugs(base, monkeypatch, plant):
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "bare_outline", plant(bare_outline))
    with pytest.raises(AssertionError):
        for source, path in grammar_cases():
            check_against_base(base, source, path)


# Differential: base module vs current over every fixture, with receipts. ------


def test_receipts_replay_and_change_only_the_named_s220_defect(receipts, base):
    fresh = receipts.build_receipts(base)
    committed = json.loads(receipts.RECEIPTS.read_text(encoding="utf-8"))
    assert receipts.committed_view(fresh) == committed
    # Every fixture of every jurisdiction is still replayed, not only NZ's.
    assert len({row["fixture"] for row in fresh["rows"]}) > 50
    assert fresh["gate_clauses_compared"] > 8000
    assert receipts.changed_rows(fresh) == receipts.EXPECTED_CHANGES
    for row in fresh["rows"]:
        if (row["fixture"], row["corpus_citation_path"]) in receipts.EXPECTED_CHANGES:
            assert row["base_outline"] is None
            assert row["base_recall"] == [1, 1, 2, 28, 1, 3]
            assert row["recall"] == [1, 28, 1, 3]
