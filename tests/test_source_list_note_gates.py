"""A NOTE paragraph between two list items does not own the items after it.

Manuals put a ``NOTE`` between two items of a list:

    The following storage costs are allowable:
    ● Locker fees ...
    NOTE When the locker fee covers several months, spread the fee ...
    ● Flood insurance ...

The note ends with a full stop and the next physical line is a bullet, so the
clause of the note ran on through the later items, and the gates-collapsed
check read each later item as one more simultaneous fact of the note's
``When``. The later items belong to the chapeau before the note. Only this
physical shape ends the note's clause at its own paragraph; every other shape
is read as before. All text here is synthetic.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _list_interjected_note_end,
    analyze_complete_source_unit,
)

CITATION = "us-zz/manual/synthetic/storage-costs/block-1"
CHAPEAU = "The following storage costs are allowable:"
FIRST = "● Locker fees for the storage unit itself when billed apart from the lease."
NOTE = (
    "NOTE When the locker fee covers several months, spread the fee over the "
    "months that the bill covers."
)
LATER = (
    "● Flood insurance for the storage unit itself.",
    "● Shed insurance for the structure itself.",
    "● Any portion of liability insurance that covers the structure.",
)
TAIL = "The following storage costs are not allowable:\n\n● Furniture insurance"
EXCERPT = NOTE.removeprefix("NOTE ")


def _source(
    *,
    chapeau: str = CHAPEAU,
    first: str | None = FIRST,
    note: str = NOTE,
    later: tuple[str, ...] = LATER,
    lead: str | None = None,
) -> str:
    parts = [lead, chapeau, first, note, *later, TAIL]
    return "\n\n".join(part for part in parts if part)


def _candidate(excerpt: str = EXCERPT) -> str:
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "proof_validation": {"required": True},
                "source_verification": {"corpus_citation_path": CITATION},
                "summary": "Synthetic storage costs.",
            },
            "inputs": [
                {
                    "name": "locker_fee_covers_several_months",
                    "entity": "Household",
                    "dtype": "Boolean",
                    "period": "Month",
                },
            ],
            "rules": [
                {
                    "name": "locker_fee_is_spread",
                    "kind": "derived",
                    "entity": "Household",
                    "dtype": "Judgment",
                    "period": "Month",
                    "source": "Synthetic storage costs, block 1",
                    "metadata": {
                        "proof": {
                            "atoms": [
                                {
                                    "path": "versions[0].formula",
                                    "kind": "condition",
                                    "source": {
                                        "corpus_citation_path": CITATION,
                                        "excerpt": excerpt,
                                    },
                                }
                            ]
                        }
                    },
                    "versions": [
                        {
                            "effective_from": "2026-01-01",
                            "formula": "locker_fee_covers_several_months",
                        }
                    ],
                }
            ],
        },
        sort_keys=False,
    )


def _collapsed(source: str, excerpt: str = EXCERPT) -> list:
    profile = vp._numeric_profile_for_citation_path(CITATION)
    analysis = analyze_complete_source_unit(
        _candidate(excerpt),
        source,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text,
            profile=profile,
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            vp.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    return [
        issue
        for issue in analysis.issues
        if getattr(issue, "check", None)
        == CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED.id
    ]


def _note_end(source: str, note: str = NOTE) -> int | None:
    start = source.index(note)
    return _list_interjected_note_end(source, start, len(source))


def test_note_between_list_items_ends_at_its_own_paragraph():
    source = _source()
    assert _note_end(source) == source.index(NOTE) + len(NOTE)
    assert _collapsed(source) == []


def test_second_sentence_of_the_note_ends_at_the_same_place():
    note = NOTE.replace(
        "NOTE ", "NOTE When the locker fee is paid once, allow it when billed. "
    )
    source = _source(note=note)
    assert _collapsed(source) == []


def test_facts_inside_the_note_are_still_counted():
    note = NOTE.replace(
        "covers several months,",
        "covers several months and the member is insured and the member is employed,",
    )
    source = _source(note=note)
    excerpt = note.removeprefix("NOTE ")
    issues = _collapsed(source, excerpt)
    assert len(issues) == 1
    message = str(issues[0])
    # The finding quotes the note alone, and counts only its own facts.
    assert 'bill covers." states 2 facts' in message
    assert "insurance, member" in message
    assert "flood" not in message.casefold()
    assert "structure" not in message.casefold()


CONTROLS = {
    # A real list of conditions under a cue: every bullet is a fact.
    "conjunctive-list-under-when": (
        _source(
            chapeau="The locker fee is spread when all of the following are met:",
            first="● The locker fee covers several months.",
            note="● The member has insurance for the storage unit.",
        ),
        "The locker fee is spread when all of the following are met:",
    ),
    # A When sentence followed by bullets, with no list open before it.
    "when-sentence-then-bullets-no-chapeau": (
        _source(chapeau="", first=None, note=EXCERPT),
        EXCERPT,
    ),
    "note-then-bullets-no-list-before": (
        _source(chapeau="", first=None),
        EXCERPT,
    ),
    "note-after-plain-paragraph": (
        _source(first="Locker fees are storage costs."),
        EXCERPT,
    ),
    "note-after-chapeau-without-item": (_source(first=None), EXCERPT),
    "note-after-lead-and-chapeau-without-item": (
        _source(lead="Storage costs are shelter costs.", first=None),
        EXCERPT,
    ),
    # The note itself introduces the bullets.
    "note-ends-with-colon": (
        _source(note=NOTE.replace("covers.", "covers, for the following:")),
        EXCERPT.replace("covers.", "covers, for the following:"),
    ),
    "note-is-the-conditional-chapeau": (
        _source(
            note="NOTE When the locker fee covers several months, the following apply:"
        ),
        "When the locker fee covers several months, the following apply:",
    ),
    "note-refers-forward-these": (
        _source(note=NOTE.replace("the months that", "these months that")),
        EXCERPT.replace("the months that", "these months that"),
    ),
    "note-refers-forward-all": (
        _source(note=NOTE.replace("the months that", "all months that")),
        EXCERPT.replace("the months that", "all months that"),
    ),
    "note-refers-forward-below": (
        _source(note=NOTE.replace("covers.", "covers below.")),
        EXCERPT.replace("covers.", "covers below."),
    ),
    "note-holds-a-semicolon": (
        _source(note=NOTE.replace("months, spread", "months; spread")),
        EXCERPT.replace("months, spread", "months; spread"),
    ),
    "note-without-full-stop": (
        _source(note=NOTE.removesuffix(".")),
        EXCERPT.removesuffix("."),
    ),
    "note-on-two-lines": (
        _source(note=NOTE.replace(" spread", "\nspread")),
        EXCERPT.replace(" spread", "\nspread"),
    ),
    "note-indented": (_source(note="  " + NOTE), EXCERPT),
    "note-lower-case": (_source(note=NOTE.replace("NOTE", "Note")), EXCERPT),
    "plain-when-paragraph-in-list": (_source(note=EXCERPT), EXCERPT),
    # The chapeau before the note is conditional or states requirements.
    "chapeau-conditional-when": (
        _source(
            chapeau="The following storage costs are allowable when the unit is leased:"
        ),
        EXCERPT,
    ),
    "chapeau-conditional-only-if": (
        _source(chapeau="Allow the following storage costs only if billed:"),
        EXCERPT,
    ),
    "chapeau-all-of-the-following": (
        _source(chapeau="A storage cost has all of the following:"),
        EXCERPT,
    ),
    "chapeau-must-have": (
        _source(chapeau="A storage cost must have the following:"),
        EXCERPT,
    ),
    "chapeau-meets": (
        _source(chapeau="A storage cost meets the following:"),
        EXCERPT,
    ),
    "chapeau-indented": (
        _source(lead="Storage costs are shelter costs.", chapeau="  " + CHAPEAU),
        EXCERPT,
    ),
    "chapeau-is-a-bullet-with-another-marker": (
        _source(chapeau="• " + CHAPEAU),
        EXCERPT,
    ),
    "chapeau-is-a-note": (_source(chapeau="NOTE " + CHAPEAU), EXCERPT),
    "chapeau-requirements": (
        _source(chapeau="The following requirements are for storage costs:"),
        EXCERPT,
    ),
    "chapeau-earlier-sentence-conditional": (
        _source(
            chapeau="Allow a cost when it is billed. " + CHAPEAU,
        ),
        EXCERPT,
    ),
    "chapeau-without-following": (
        _source(chapeau="Allowable storage costs:"),
        EXCERPT,
    ),
    "chapeau-without-colon": (
        _source(chapeau="The following storage costs are allowable."),
        EXCERPT,
    ),
    "chapeau-is-a-bullet": (
        _source(chapeau="● " + CHAPEAU),
        EXCERPT,
    ),
    "chapeau-on-two-lines": (
        _source(chapeau=CHAPEAU.replace(" storage", "\nstorage")),
        EXCERPT,
    ),
    # Nested lists and other markers.
    "item-before-note-opens-its-own-list": (
        _source(first="● Locker fees for the storage unit in these cases:"),
        EXCERPT,
    ),
    "earlier-item-opens-its-own-list": (
        _source(first="● Locker fees for the storage unit in these cases:\n\n" + FIRST),
        EXCERPT,
    ),
    "item-before-note-indented": (_source(first="  " + FIRST), EXCERPT),
    "item-before-note-is-unmarked-sub-item": (
        _source(first=FIRST + "\n\nLocker fees billed by a depot"),
        EXCERPT,
    ),
    "item-after-note-indented": (
        _source(later=("  " + LATER[0], *LATER[1:])),
        EXCERPT,
    ),
    "item-after-note-other-marker": (
        _source(later=(LATER[0].replace("●", "•"), *LATER[1:])),
        EXCERPT,
    ),
    "item-after-note-continues-the-note": (
        _source(later=("● And flood insurance is billed.", *LATER[1:])),
        EXCERPT,
    ),
    "item-after-note-is-a-condition": (
        _source(later=("● If flood insurance is billed.", *LATER[1:])),
        EXCERPT,
    ),
    "item-after-note-lower-case": (
        _source(later=("● flood insurance for the storage unit itself.", *LATER[1:])),
        EXCERPT,
    ),
    "carriage-returns": (_source().replace("\n", "\r\n"), EXCERPT),
}


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_every_other_shape_stays_refused(name):
    source, excerpt = CONTROLS[name]
    assert len(_collapsed(source, excerpt)) == 1


@pytest.mark.parametrize(
    "name",
    sorted(
        set(CONTROLS)
        - {
            "conjunctive-list-under-when",
            "when-sentence-then-bullets-no-chapeau",
            "plain-when-paragraph-in-list",
            "note-is-the-conditional-chapeau",
        }
    ),
)
def test_no_other_shape_is_a_list_note(name):
    source, _excerpt = CONTROLS[name]
    start = source.index("NOTE") if "NOTE" in source else source.index("Note")
    assert _list_interjected_note_end(source, start, len(source)) is None


@pytest.mark.parametrize(
    "word",
    "following follows below next subsequent these those list lists listed item "
    "items bullet bullets either both all any each every".split()
    + ["one of", ":", ";"],
)
def test_note_with_a_forward_reference_is_not_a_list_note(word):
    note = NOTE.replace("covers.", f"covers ({word}).")
    assert _note_end(_source(note=note), note) is None


@pytest.mark.parametrize(
    "word",
    "if when whenever unless provided only except until while".split(),
)
def test_chapeau_line_with_a_condition_cue_is_not_a_list_chapeau(word):
    chapeau = f"Storage costs ({word}) are billed. " + CHAPEAU
    assert _note_end(_source(chapeau=chapeau)) is None


@pytest.mark.parametrize(
    "word",
    "all each every both must shall require requirements conditions criteria "
    "meet meets met qualify qualifying eligible eligibility apply applies applied "
    "circumstance circumstances situation situations occur occurs".split(),
)
def test_chapeau_sentence_with_a_requirement_word_is_not_a_list_chapeau(word):
    chapeau = f"The following storage costs ({word}) are allowable:"
    assert _note_end(_source(chapeau=chapeau)) is None


@pytest.mark.parametrize(
    "word", "And Or Nor But If When Unless Provided Only Except".split()
)
def test_item_after_the_note_that_starts_with_a_connector_is_not_a_new_item(word):
    later = (f"● {word} flood insurance is billed.", *LATER[1:])
    assert _note_end(_source(later=later)) is None


def test_requirement_words_before_the_chapeau_sentence_do_not_count():
    chapeau = (
        "To get the cost, the unit must be leased. (See Leases for the "
        "criteria.) Storage costs include the following:"
    )
    source = _source(chapeau=chapeau)
    assert _note_end(source) is not None
    assert _collapsed(source) == []


def test_clause_that_ends_inside_the_note_is_not_cut():
    source = _source()
    start = source.index(NOTE)
    assert _list_interjected_note_end(source, start, start + len(NOTE)) is None
    assert _list_interjected_note_end(source, start, start + len(NOTE) + 1) == (
        start + len(NOTE)
    )


def test_clause_that_starts_outside_the_note_is_not_cut():
    source = _source()
    assert _list_interjected_note_end(source, 0, len(source)) is None
    after = source.index(LATER[0])
    assert _list_interjected_note_end(source, after, len(source)) is None
    note_end = source.index(NOTE) + len(NOTE)
    assert _list_interjected_note_end(source, note_end, len(source)) is None
    assert _list_interjected_note_end(source, note_end - 1, len(source)) == note_end
    before = source.index(NOTE) - 1
    assert _list_interjected_note_end(source, before, len(source)) is None


def test_several_items_before_the_note_reach_the_chapeau():
    source = _source(first=FIRST + "\n\n● Gate fees for the storage unit.")
    assert _note_end(source) is not None
    assert _collapsed(source) == []


def test_last_note_of_the_text_has_no_following_item():
    source = "\n\n".join([CHAPEAU, FIRST, NOTE])
    assert _note_end(source) is None
