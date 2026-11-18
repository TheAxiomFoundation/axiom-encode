"""A NOTE paragraph between two list items does not own the items after it.

Manuals put a ``NOTE`` between two items of a list:

    The following storage expenses are allowable:
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

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _list_interjected_note_end,
    analyze_complete_source_unit,
)

CITATION = "us-zz/manual/synthetic/storage-costs/block-1"
CHAPEAU = "The following storage expenses are allowable:"
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
TAIL = "The following storage expenses are not allowable:\n\n● Furniture insurance"
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


def _candidate(excerpt: str | list[str] = EXCERPT) -> str:
    excerpts = [excerpt] if isinstance(excerpt, str) else excerpt
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "proof_validation": {"required": True},
                "source_verification": {"corpus_citation_path": CITATION},
                "summary": "Synthetic storage expenses.",
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
                    "source": "Synthetic storage expenses, block 1",
                    "metadata": {
                        "proof": {
                            "atoms": [
                                {
                                    "path": "versions[0].formula",
                                    "kind": "condition",
                                    "source": {
                                        "corpus_citation_path": CITATION,
                                        "excerpt": text,
                                    },
                                }
                                for text in excerpts
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


def _collapsed(source: str, excerpt: str | list[str] = EXCERPT) -> list:
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
        _source(first="Locker fees are storage expenses."),
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
            chapeau="The following storage expenses are allowable when the unit is leased:"
        ),
        EXCERPT,
    ),
    "chapeau-conditional-only-if": (
        _source(chapeau="Allow the following storage expenses only if billed:"),
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
        _source(chapeau="The following requirements are for storage expenses:"),
        EXCERPT,
    ),
    "chapeau-earlier-sentence-conditional": (
        _source(
            chapeau="Allow a cost when it is billed. " + CHAPEAU,
        ),
        EXCERPT,
    ),
    "chapeau-without-following": (
        _source(chapeau="Allowable storage expenses:"),
        EXCERPT,
    ),
    "chapeau-without-colon": (
        _source(chapeau="The following storage expenses are allowable."),
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


@pytest.mark.parametrize("word", sorted(completeness._LIST_NOTE_FORWARD_WORDS))
@pytest.mark.parametrize("place", ["middle", "end"])
def test_note_with_a_forward_word_anywhere_is_not_a_list_note(word, place):
    if place == "middle":
        note = NOTE.replace("spread the fee", f"spread the {word} fee")
    else:
        note = NOTE.replace("bill covers.", f"bill covers the {word} storage.")
    assert note != NOTE
    assert _note_end(_source(note=note), note) is None
    upper = note.replace(f" {word} ", f" {word.upper()} ")
    assert upper != note
    assert _note_end(_source(note=upper), upper) is None


@pytest.mark.parametrize("mark", [":", ";", " one of the fees"])
def test_note_with_a_colon_semicolon_or_one_of_is_not_a_list_note(mark):
    note = NOTE.replace("months, spread", f"months{mark}, spread")
    assert _note_end(_source(note=note), note) is None


@pytest.mark.parametrize("word", sorted(completeness._LIST_NOTE_CLOSED_CLASS_WORDS))
def test_closed_class_word_is_refused_at_the_note_end_and_in_the_chapeau(word):
    note = NOTE.replace("bill covers.", f"bill covers {word}.")
    assert _note_end(_source(note=note), note) is None
    upper = NOTE.replace("bill covers.", f"bill covers {word.upper()}.")
    assert _note_end(_source(note=upper), upper) is None
    for chapeau in (
        f"The following {word} expenses are allowable:",
        f"The following payments are allowable {word} expenses:",
        f"Allow {word} expenses for the leased unit, including the following:",
        f"Allow storage expenses for the {word} unit, including the following:",
    ):
        assert _note_end(_source(chapeau=chapeau)) is None
        assert _note_end(_source(chapeau=chapeau.replace(word, word.upper()))) is None


@pytest.mark.parametrize("word", sorted(completeness._LIST_NOTE_REQUIREMENT_WORDS))
def test_requirement_word_in_the_chapeau_is_refused(word):
    for chapeau in (
        f"The following {word} expenses are allowable:",
        f"The following payments are allowable {word} expenses:",
        f"Allow {word} expenses for the leased unit, including the following:",
        f"Allow storage expenses for the {word} unit, including the following:",
    ):
        assert _note_end(_source(chapeau=chapeau)) is None
        assert _note_end(_source(chapeau=chapeau.replace(word, word.title()))) is None


def test_words_of_the_lists_are_what_blocks_those_chapeaus():
    """The same shapes with a neutral word are accepted."""
    for chapeau in (
        "The following rental expenses are allowable:",
        "The following payments are allowable rental expenses:",
        "Allow rental expenses for the leased unit, including the following:",
        "Allow storage expenses for the rental unit, including the following:",
    ):
        assert _note_end(_source(chapeau=chapeau)) is not None
    note = NOTE.replace("bill covers.", "bill covers rental.")
    assert _note_end(_source(note=note), note) is not None
    note = NOTE.replace("spread the fee", "spread the rental fee")
    assert _note_end(_source(note=note), note) is not None


@pytest.mark.parametrize(
    "chapeau",
    [
        CHAPEAU,
        "The following expenses are allowable:",
        "The following payments are allowable:",
        "The following storage unit expenses are allowable:",
        "The following payments are allowable storage expenses:",
        "The following payments are allowable storage unit expenses:",
        "Allow storage expenses for the leased unit, including the following:",
        "Allow expenses for the unit, including the following:",
        "Allow rental storage expenses for the primary leased unit, including the following:",
    ],
)
def test_listed_enumeration_chapeaus_are_accepted(chapeau):
    source = _source(chapeau=chapeau)
    assert _note_end(source) is not None
    assert _collapsed(source) == []


# Round 1 review: chapeaus of joint requirements that two deny lists let pass.
JOINT_REQUIREMENT_CHAPEAUS = [
    "The household needs the following to get the storage deduction:",
    "The household gets the deduction where it has the following:",
    "The household gets the deduction as long as it has the following:",
    "The household gets the deduction once it has the following:",
    "The household gets the deduction after it shows the following:",
    "The household gets the deduction in case of the following:",
    "The household gets the deduction given the following:",
    "The household gets the deduction with the following:",
    "The following are needed:",
    "The following are necessary:",
    "The following are prerequisites:",
    "The following are mandatory:",
    "The household has to show the following:",
    "The household should have the following:",
    "The following tests are satisfied:",
    "The following two tests are used:",
    "The deduction rests on the following three facts:",
    "The household provides proof of the following:",
    "The household demonstrates the following:",
    "The following factors are present together:",
    # Abbreviation full stops used to cut the sentence before its requirement.
    "The applicant must meet, under A.R.S. 46-292, the following:",
    "All of the tests in 7 C.F.R. 273.9 and the following:",
    "Each household must meet (see Sec. 4) the following:",
    "A household must show, e.g. for a storage unit, the following:",
    # A requirement in an earlier sentence of the chapeau line.
    "The household must have these to get the cost. They include the following:",
    "To get the cost, the unit must be leased. Storage costs include the following:",
    "The following tests are used:",
    # Round 2 review: joint wording that fitted the first allow-list.
    "The following are allowed together:",
    "The following are allowed jointly:",
    "The following are allowed combined:",
    "The following combination is allowable:",
    "The following set is allowed:",
    "The following expenses are allowed as joint proof:",
    "The following necessary proofs are allowed:",
    "The following mandatory proofs are allowed:",
    "The following needed proofs are allowed:",
    "The following prerequisites are allowable:",
    "The following expenses are allowable together:",
    "The following expenses are allowable jointly:",
    "The following joint expenses are allowable:",
    "The following expenses are allowable as joint expenses:",
    # Cues inside a slash or hyphen token, and open-class cue words.
    "The following if/when expenses are allowable:",
    "The following where/when expenses are allowable:",
    "The following once/when expenses are allowable:",
    "The following only/when expenses are allowable:",
    "The following when-leased expenses are allowable:",
    "The following court-ordered expenses are allowable:",
    "The following wherever expenses are allowable:",
    "The following given expenses are allowable:",
    "The following assuming expenses are allowable:",
    "The following absent expenses are allowable:",
    # Forms of the first allow-list that no accepted real note needs.
    "The following are allowable storage expenses:",
    "The following are allowable:",
    "The following are not allowed as storage expenses:",
    "The following storage expenses are not allowable:",
    "The following storage expense is allowable:",
    "The following storage expenses are allowed:",
    "The following storage costs are allowable:",
    "The following big storage unit expenses are allowable:",
    "The following payments are allowable big storage unit expenses:",
    "The following payments are allowable storage proof:",
    "The following payments are allowable expenses expenses expenses expenses:",
    "Allow storage costs for the leased unit, including the following:",
    "Allow storage expenses for a leased unit, including the following:",
    "Allow storage expenses for the, including the following:",
    "Allow storage expenses for the big old leased storage unit, including the following:",
    "Allow big old storage expenses for the leased unit, including the following:",
    "Allow storage expenses for the leased unit including the following:",
    "Allow storage expenses for the leased unit, including the following",
    # Forms next to the listed ones.
    "The Following storage expenses are allowable:",
    "the following storage expenses are allowable:",
    "The following storage expenses are allowable",
    "The following storage expenses are allowable: ",
    "The following storage expenses are payable:",
    "The following storage expenses were allowable:",
    "The following storage expenses are allowable when leased:",
    "The following storage expenses that are leased are allowable:",
    "The following storage expenses are allowable for those who have a lease:",
    "The following storage expenses (leased) are allowable:",
    "The following storage expenses for the unit are allowable:",
    "The following costs of storage are allowable:",
    "The following are allowable costs for the unit:",
    "Following storage expenses are allowable:",
    "Storage costs include the following:",
    "Allow storage expenses including the following:",
    "Allow storage expenses that are leased, including the following:",
    "Allow, including the following:",
    "Disallow storage expenses, including the following:",
]


@pytest.mark.parametrize("chapeau", JOINT_REQUIREMENT_CHAPEAUS)
def test_any_other_chapeau_keeps_the_later_items_as_facts(chapeau):
    source = _source(
        chapeau=chapeau,
        first="● The household leases the storage unit.",
        later=(
            "● The member is insured for the unit.",
            "● The member is employed by the depot.",
        ),
    )
    assert _note_end(source) is None
    assert len(_collapsed(source)) == 1


def test_requirement_in_an_earlier_paragraph_with_another_chapeau_is_refused():
    source = _source(
        lead="A household is eligible when it meets every test in this section.",
        chapeau="The following tests are used:",
    )
    assert _note_end(source) is None
    assert len(_collapsed(source)) == 1


@pytest.mark.parametrize(
    "note",
    [
        "NOTE Spread the locker fee when the household has the remaining entries.",
        "NOTE Spread the locker fee when the household also has.",
        "NOTE This applies only when the client is.",
        "NOTE When the fee covers several months and.",
        "NOTE Spread the locker fee when the household has the rest.",
        "NOTE Spread the locker fee when the household has the other ones.",
        "NOTE Spread the locker fee when the member is insured for.",
        "NOTE Spread the locker fee when the member is insured and the.",
        "NOTE Spread the locker fee when the member is insured only.",
        "NOTE Spread the locker fee when the household has the remaining costs.",
        "NOTE Spread the locker fee when the household has the other costs.",
        "NOTE Spread the locker fee when the household has the additional costs.",
        "NOTE Spread the locker fee when the household has the further costs.",
        "NOTE Spread the locker fee when the household has such costs.",
        "NOTE Spread the locker fee for the costs that come after this note.",
        "NOTE Spread the locker fee when the household has the costs shown.",
        "NOTE Spread the locker fee when the household has the costs shown here.",
        "NOTE Spread the locker fee when the household has the costs as shown.",
        "NOTE Spread the locker fee when the household passes the tests described.",
        "NOTE Spread the locker fee when the household passes the tests given.",
        "NOTE Spread the locker fee when the household has the costs named hereafter.",
        "NOTE Spread the locker fee when the household has the costs set out later.",
        "NOTE Spread the locker fee when the household has the two additional costs.",
        "NOTE Spread the locker fee when the member is insured also.",
        "NOTE Spread the locker fee when the member has the remaining.",
        "NOTE Spread the locker fee when the member insured IS.",
        "NOTE Spread the locker fee when the member is insured For.",
        "NOTE Spread the locker fee when the member is insured (unit).",
        "NOTE Spread the locker fee when the member is insured in 2026.",
    ],
)
def test_note_that_does_not_end_on_its_own_content_is_not_a_list_note(note):
    source = _source(note=note)
    assert _note_end(source, note) is None
    excerpt = note.removeprefix("NOTE ")
    if "insured" in note or "has the" in note:
        assert len(_collapsed(source, excerpt)) == 1


@pytest.mark.parametrize(
    "head",
    ["NOTE.", "NOTE -", "NOTE\t", "NOTES", "NOTE:", "NOTE  ", "NOTE w", "NOTE 2"],
)
def test_only_the_documented_note_head_is_a_note(head):
    note = head.rstrip("w2") + NOTE.removeprefix("NOTE")
    if head.endswith(("w", "2")):
        note = "NOTE " + head[-1] + NOTE.removeprefix("NOTE W")
    assert _note_end(_source(note=note), note) is None


CONNECTORS = [
    connector.capitalize() for connector in completeness._LIST_NOTE_CONNECTORS
]


@pytest.mark.parametrize("word", CONNECTORS)
@pytest.mark.parametrize("place", ["next", "last", "before", "unmarked"])
def test_connector_anywhere_in_the_list_keeps_the_old_reading(word, place):
    item = f"● {word} the member is insured for the unit."
    if place == "next":
        source = _source(later=(item, *LATER[1:]))
    elif place == "last":
        source = _source(later=(*LATER, item))
    elif place == "before":
        source = _source(first=item + "\n\n" + FIRST)
    else:
        source = _source(later=(*LATER, item.removeprefix("● ")))
    assert _note_end(source) is None


@pytest.mark.parametrize(
    "later",
    [
        (*LATER, "● and the member is insured."),
        (*LATER, "and the member is insured for the unit."),
        (LATER[0].removesuffix(".") + ", and", *LATER[1:]),
        (LATER[0], LATER[1].removesuffix(".") + ", or", LATER[2]),
        (LATER[0], LATER[1].removesuffix(".") + ";", LATER[2]),
        (LATER[0], LATER[1].removesuffix(".") + " and", LATER[2]),
        (LATER[0], LATER[1].removesuffix(".") + " or", LATER[2]),
        (LATER[0], LATER[1].removesuffix(".") + ",", LATER[2]),
        (*LATER, "●Gate insurance for the unit."),
        (*LATER, "●  Gate insurance for the unit."),
        (*LATER, "• Gate insurance for the unit."),
        (*LATER, "  ● Gate insurance for the unit."),
        (*LATER, "● Gate insurance\nfor the unit."),
        (*LATER, "Gate insurance\nfor the unit."),
    ],
)
def test_every_later_paragraph_of_the_list_is_checked(later):
    source = _source(later=later)
    assert _note_end(source) is None
    assert len(_collapsed(source)) == 1


@pytest.mark.parametrize(
    "first",
    [
        "• Gate fees for the unit.\n\n" + FIRST,
        "●Gate fees for the unit.\n\n" + FIRST,
        "● gate fees for the unit.\n\n" + FIRST,
        "● Gate fees for the unit, and\n\n" + FIRST,
        "● Gate fees for the unit;\n\n" + FIRST,
        "● Gate fees for the unit:\n\n" + FIRST,
        FIRST.removesuffix(".") + ":",
        FIRST.removesuffix(".") + ", or",
        "●" + FIRST[2:],
    ],
)
def test_every_item_before_the_note_is_checked(first):
    assert _note_end(_source(first=first)) is None


def test_paragraph_after_the_note_must_be_an_item():
    source = _source(later=("Flood insurance is a storage cost.", *LATER))
    assert _note_end(source) is None


@pytest.mark.parametrize("marker", ["-", "*", "X", "1.", "o"])
def test_other_list_markers_are_not_items(marker):
    source = _source().replace("●", marker)
    assert _note_end(source) is None


def test_list_shapes_that_are_still_accepted():
    # The list at the very start of the text, and the item after the note as
    # the last paragraph of the text.
    source = "\n\n".join([CHAPEAU, FIRST, NOTE, LATER[0]])
    assert source.startswith(CHAPEAU)
    assert _note_end(source) == source.index(NOTE) + len(NOTE)
    assert _collapsed(source + "\n\n" + LATER[1]) == []
    # Blank lines that hold spaces or tabs.
    spaced = _source().replace("\n\n", "\n \t\n")
    assert _note_end(spaced) == spaced.index(NOTE) + len(NOTE)
    # An item that starts with a digit.
    digit = _source(later=("● 24-hour gate insurance for the unit.", *LATER))
    assert _note_end(digit) is not None
    # A later note, or any fresh paragraph, ends the list.
    later_note = _source(
        later=(LATER[0], "NOTE Count gate insurance in the month of the bill.")
    )
    assert _note_end(later_note) is not None
    # Text after the end of the list is not examined.
    ended = _source() + "\n\nand other costs that the worker lists."
    assert _note_end(ended) is not None


@pytest.mark.parametrize(
    "later",
    [
        # A later item that opens a list of its own.
        ("● Flood insurance when all of the following are met:", *LATER[1:]),
        (LATER[0], "● Other insurance, which includes any of the following:"),
        # An unmarked line that continues the list.
        (LATER[0], "and when the following are true:", *LATER[1:]),
        (LATER[0], "And when the following are true:", *LATER[1:]),
        (LATER[0], "1. Gate insurance for the unit."),
        (LATER[0], "A. Gate insurance for the unit."),
        (LATER[0], "a) Gate insurance for the unit."),
        (LATER[0], "(a) Gate insurance for the unit."),
        (LATER[0], "24 hour gate insurance for the unit."),
        (LATER[0], "gate insurance for the unit."),
        (LATER[0], "Gate insurance for the unit, and"),
    ],
)
def test_list_after_the_note_is_a_plain_run_of_items(later):
    assert _note_end(_source(later=later)) is None


def test_carriage_return_line_endings_are_refused():
    assert _note_end(_source().replace("\n", "\r\n")) is None
    assert _note_end(_source().replace("\n", "\r")) is None


def _same_as_main(monkeypatch, source, excerpts):
    """Findings with the rule, and with the rule switched off (as on main)."""
    with_rule = [str(issue) for issue in _collapsed(source, excerpts)]
    with monkeypatch.context() as patch:
        patch.setattr(completeness, "_list_interjected_note_end", lambda *_: None)
        as_main = [str(issue) for issue in _collapsed(source, excerpts)]
    return with_rule, as_main


BUSY_ITEM = (
    "● Flood insurance when the unit is leased and the member is insured and "
    "the member is employed."
)
BUSY = _source(later=(BUSY_ITEM, *LATER[1:]))
PLAIN_NOTE = "NOTE Spread the locker fee over the months that the bill covers."


@pytest.mark.parametrize(
    "source,excerpts",
    [
        # A later item alone.
        (BUSY, [BUSY_ITEM.removeprefix("● ")]),
        (_source(), [LATER[0].removeprefix("● ")]),
        # One excerpt that spans two later items.
        (BUSY, [BUSY_ITEM.removeprefix("● ") + "\n\n" + LATER[1]]),
        (_source(), [LATER[0].removeprefix("● ") + "\n\n" + LATER[1]]),
        # One excerpt that starts in the note and runs into the items.
        (_source(), [EXCERPT + "\n\n" + LATER[0]]),
        (BUSY, [EXCERPT + "\n\n" + BUSY_ITEM]),
        # The chapeau, and the item before the note.
        (_source(), [CHAPEAU]),
        (_source(), [FIRST.removeprefix("● ")]),
        (_source(), [CHAPEAU, FIRST.removeprefix("● "), LATER[0].removeprefix("● ")]),
        # Round 2 input 1: a later item that is a conditional chapeau.
        (
            _source(
                note=PLAIN_NOTE,
                later=(
                    "● Flood insurance when all of the following are met:",
                    "● The member is insured for the unit.",
                    "● The member is employed by the depot.",
                ),
            ),
            ["Flood insurance when all of the following are met:"],
        ),
        # Round 2 input 3: an unmarked colon line after a later item.
        (
            _source(
                note=PLAIN_NOTE,
                later=(
                    LATER[0],
                    "and when the following are true:",
                    "● The member is insured for the unit.",
                    "● The member is employed by the depot.",
                ),
            ),
            ["and when the following are true:"],
        ),
    ],
)
def test_proof_that_is_not_the_note_is_read_as_on_main(monkeypatch, source, excerpts):
    with_rule, as_main = _same_as_main(monkeypatch, source, excerpts)
    assert with_rule == as_main


def test_proofs_that_reach_past_the_note_are_refused_like_on_main(monkeypatch):
    """The three cases above that matter: main has a finding, so has the rule."""
    for excerpts in (
        [BUSY_ITEM.removeprefix("● ") + "\n\n" + LATER[1]],
        [EXCERPT + "\n\n" + BUSY_ITEM],
        [EXCERPT + "\n\n" + LATER[0]],
    ):
        source = BUSY if "leased" in excerpts[0] else _source()
        with_rule, as_main = _same_as_main(monkeypatch, source, excerpts)
        assert with_rule == as_main
        assert len(with_rule) == 1


def test_all_atoms_only_the_note_atom_is_read_from_the_note(monkeypatch):
    atoms = [
        CHAPEAU,
        FIRST.removeprefix("● "),
        EXCERPT,
        *(item.removeprefix("● ") for item in LATER),
    ]
    with_rule, as_main = _same_as_main(monkeypatch, _source(), atoms)
    assert len(as_main) == 1 and with_rule == []
    # Without the note atom nothing differs from main.
    others = [atom for atom in atoms if atom != EXCERPT]
    with_rule, as_main = _same_as_main(monkeypatch, _source(), others)
    assert with_rule == as_main
    # A later item with facts of its own is still demanded through its atom.
    busy_atoms = [
        CHAPEAU,
        FIRST.removeprefix("● "),
        EXCERPT,
        BUSY_ITEM.removeprefix("● "),
        *(item.removeprefix("● ") for item in LATER[1:]),
    ]
    with_rule, as_main = _same_as_main(monkeypatch, BUSY, busy_atoms)
    assert len(with_rule) == 1
    assert "Flood insurance when the unit is leased" in with_rule[0]
    assert "NOTE" not in with_rule[0]


@pytest.mark.parametrize("order", [0, 1])
def test_second_proof_of_the_same_clause_keeps_every_gate(monkeypatch, order):
    """A note proof and a run-on proof in one rule: the run-on reading wins."""
    note = NOTE.replace(
        "covers several months,",
        "covers several months and the member is insured and the member is employed,",
    )
    source = _source(note=note)
    inside = note.removeprefix("NOTE ")
    run_on = "spread the fee over the months that the bill covers.\n\n" + LATER[0]
    excerpts = [inside, run_on] if order else [run_on, inside]
    with_rule, as_main = _same_as_main(monkeypatch, source, excerpts)
    assert len(as_main) == 1
    assert with_rule == as_main


def test_note_is_not_read_when_the_clause_has_no_gates(monkeypatch):
    """The rule can only remove gates that the whole clause had."""
    from axiom_encode.harness import source_completeness as completeness

    calls = []
    real = completeness._list_interjected_note_end

    def recording(source_text, start, end):
        calls.append((start, end))
        return real(source_text, start, end)

    monkeypatch.setattr(completeness, "_list_interjected_note_end", recording)
    plain = _source(later=("● Flood cover.", "● Shed cover."))
    assert _note_end(plain) is not None
    assert _collapsed(plain) == []
    assert calls == []
    assert _collapsed(_source()) == []
    assert len(calls) == 1


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
