"""Structural reading of bullet lists in the gates-collapsed check.

The reading is behind ``source_list_structure.STRUCTURAL_GATE_READING_ENABLED``
(default off). Every source text here is constructed; see
``docs/gates-structural-reading.md`` for the design and its four invariants:

(a) fail closed: an unrecognised layout or chapeau keeps the flat reading;
(b) a list of alternatives is still verified, on the value of the formula;
(c) a rule that quotes only items of a list is read against the whole list;
(d) a conjunctive list is protected at least as on the flat reading.
"""

from __future__ import annotations

import functools
import itertools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_list_structure as structure
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    analyze_complete_source_unit,
)

CITATION = "us-zz/manual/test/structural/block-1"


def _candidate(inputs, rules, dtype="Judgment"):
    """Build a candidate: ``rules`` are (name, formula, proof excerpts)."""

    document = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": name, "entity": "Household", "dtype": "Boolean", "period": "Month"}
            if isinstance(name, str)
            else name
            for name in inputs
        ],
        "rules": [
            {
                "name": name,
                "kind": "derived",
                "entity": "Household",
                "dtype": dtype,
                "period": "Month",
                "source": "Test manual, block-1",
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
                            for excerpt in excerpts
                        ]
                    }
                },
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            }
            for name, formula, excerpts in rules
        ],
    }
    return yaml.safe_dump(document, sort_keys=False)


def _findings(content, source, *, on, monkeypatch):
    monkeypatch.setattr(structure, "STRUCTURAL_GATE_READING_ENABLED", on)
    profile = vp._numeric_profile_for_citation_path(CITATION)
    analysis = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            vp.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    return [
        str(issue)
        for issue in analysis.issues
        if getattr(issue, "check", None)
        == CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED.id
    ]


def _both(inputs, rules, source, monkeypatch, dtype="Judgment"):
    """Return the number of findings (switch off, switch on)."""

    content = _candidate(inputs, rules, dtype)
    return (
        len(_findings(content, source, on=False, monkeypatch=monkeypatch)),
        len(_findings(content, source, on=True, monkeypatch=monkeypatch)),
    )


ANY_CHAPEAU = "A parking permit must be renewed when any of the following occur:"
ANY_ITEMS = (
    "New vehicle",
    "Replacement permit",
    "When a change of employer is reported",
    "When the fee amount is disputed",
)
ANY = ANY_CHAPEAU + "".join(f"\n\n● {item}" for item in ANY_ITEMS)
ANY_INPUTS = [
    "new_vehicle",
    "replacement_permit",
    "change_of_employer_reported",
    "fee_amount_disputed",
]
A, B, C, D = ANY_INPUTS
FAITHFUL = f"{A} or {B} or {C} or {D}"

ALL_ITEMS = (
    "The holder owns a vehicle.",
    "The holder lives in the district.",
    "The holder has paid the fee.",
)
ALL_INPUTS = ["holder_owns_vehicle", "holder_lives_in_district", "holder_has_paid_fee"]
ALL_CHAPEAU = "A parking permit is granted if all of the following apply:"
PLAIN_CHAPEAU = "A parking permit is granted if the holder meets these conditions:"


def _list(chapeau, items):
    return chapeau + "".join(f"\n\n● {item}" for item in items)


# --- the switch ----------------------------------------------------------------


def test_the_switch_is_off_by_default():
    assert structure.STRUCTURAL_GATE_READING_ENABLED is False


def test_the_structural_code_is_not_reached_while_the_switch_is_off(monkeypatch):
    def unreachable(*_args, **_kwargs):
        raise AssertionError("structural reading ran with the switch off")

    monkeypatch.setattr(structure, "parse_source_lists", unreachable)
    monkeypatch.setattr(sc, "_structural_clause_reading", unreachable)
    monkeypatch.setattr(sc, "_alternative_list_defect", unreachable)
    content = _candidate(ANY_INPUTS, [("renewal", FAITHFUL, [ANY_CHAPEAU])])
    # The flat reading refuses the faithful candidate: that is the false
    # positive this reading removes, and it stays while the switch is off.
    assert len(_findings(content, ANY, on=False, monkeypatch=monkeypatch)) == 1
    # The same tripwires fire with the switch on, so they do guard the path.
    with pytest.raises(AssertionError, match="switch off"):
        _findings(content, ANY, on=True, monkeypatch=monkeypatch)


# --- the tree --------------------------------------------------------------------


def test_the_tree_reads_chapeau_items_notes_and_unmarked_children():
    text = (
        "Proof is needed when any of the following occur:\n\n"
        "● A new permit is requested and when any of the following occur:\n\n"
        "When the holder is new, proof of address must be shown.\n\n"
        "When the holder has unpaid fines, all fines must be verified.\n\n"
        "● The fee amount is questionable.\n\n"
        "NOTE The clerk records the reason.\n\n"
        "● The address changed.\n\n"
        "A later paragraph."
    )
    (only,) = structure.parse_source_lists(text)
    assert [text[item.start : item.end][:9] for item in only.items] == [
        "● A new p",
        "● The fee",
        "● The add",
    ]
    assert only.items_with_children == frozenset({0})
    assert text[only.end :].strip() == "A later paragraph."
    # Text follows the list, so it is alternatives but not verifiable.
    assert (only.reading, only.cue_form, only.verifiable) == (
        "alternatives",
        True,
        False,
    )


@pytest.mark.parametrize(
    "quantifier",
    [
        "any",
        "one or more",
        "either",
        "any one",
        "at least one",
        "one",
    ],
)
@pytest.mark.parametrize("cue", ["when", "if", "only when", "When"])
@pytest.mark.parametrize(
    "tail",
    ["", " occur", " occurs", " apply", " applies", " conditions are met", " is true"],
)
def test_the_closed_cue_form_is_read_as_alternatives(quantifier, cue, tail):
    chapeau = f"A permit is renewed {cue} {quantifier} of the following{tail}:"
    (only,) = structure.parse_source_lists(_list(chapeau, ANY_ITEMS))
    assert (only.reading, only.cue_form) == ("alternatives", True)


@pytest.mark.parametrize(
    "chapeau",
    [
        "Any of the following parking fees are not refundable:",
        "Any of the following can be used to prove residence:",
    ],
)
def test_the_closed_subject_form_is_read_as_alternatives_without_a_cue(chapeau):
    (only,) = structure.parse_source_lists(_list(chapeau, ANY_ITEMS))
    assert (only.reading, only.cue_form, only.binds_items) == (
        "alternatives",
        False,
        False,
    )


NOT_ALTERNATIVES = [
    ALL_CHAPEAU,
    PLAIN_CHAPEAU,
    "A permit is renewed when each one of the following occur:",
    "A permit is renewed when every one of the following occur:",
    "A permit is renewed when not any of the following occur:",
    "A permit is renewed when none of the following occur:",
    "A permit is renewed when all or any of the following occur:",
    "A permit is renewed when any of the following do not apply:",
    "A permit is renewed when any of the following never occur:",
    "A permit is renewed when any of the following fail:",
    "A permit is renewed when any two of the following occur:",
    "A permit is renewed when at least two of the following occur:",
    "A permit is renewed when any of the following occur together:",
    "A permit is renewed when any of the following and the rule below apply:",
    "A permit is renewed unless any of the following occur:",
    "A permit is renewed provided that any of the following occur:",
    "A permit is renewed where any of the following occur:",
    "A permit is renewed when the holder has any of the following:",
    "A permit is renewed when the holder lacks any of the following:",
    "The holder must not have any of the following:",
    "The following apply when any of the following occur:",
    "any of the following are not refundable:",
    "Any of the following are not refundable. All of them are listed:",
    "Each or Any of the following parking fees are not refundable:",
    "Fees are due when any of the following occur",
    "The following parking fees are not refundable:",
    "The holder must provide the following:",
]


@pytest.mark.parametrize("chapeau", NOT_ALTERNATIVES)
def test_any_other_chapeau_is_not_read_as_alternatives(chapeau):
    lists = structure.parse_source_lists(_list(chapeau, ANY_ITEMS))
    assert all(found.reading != "alternatives" for found in lists)


@pytest.mark.parametrize(
    "text",
    [
        # A marked item that ends with a colon and has marked children only.
        ANY_CHAPEAU + "\n\n● New vehicle:\n\n● Sedan\n\n● Truck\n\n● Lost card",
        # Two markers in one list.
        ANY_CHAPEAU + "\n\n● New vehicle\n\n• Lost card",
        # More items than the reader takes.
        _list(ANY_CHAPEAU, [f"Reason number {index}" for index in range(13)]),
    ],
)
def test_an_unreadable_list_layout_is_not_read_as_alternatives(text):
    assert all(
        found.reading != "alternatives" for found in structure.parse_source_lists(text)
    )


@pytest.mark.parametrize(
    "text",
    [
        ANY_CHAPEAU + "\n\n  ● New vehicle\n\n  ● Lost card",
        ANY_CHAPEAU + "\n\n●New vehicle\n\n● Lost card",
        ANY_CHAPEAU + "\n\n●  New vehicle\n\n● Lost card",
        ANY_CHAPEAU + "\n\n● New vehicle\n\n indented words\n\n● Lost card",
        ANY_CHAPEAU + "\r\n\r\n● New vehicle\r\n\r\n● Lost card",
        ANY_CHAPEAU + "\n\n● New vehicle\n\n\t● Lost card",
    ],
)
def test_an_unreadable_text_layout_gives_no_list_at_all(text):
    assert structure.parse_source_lists(text) == ()


def test_a_chapeau_that_is_itself_a_marked_item_is_not_a_list_head():
    text = (
        "A permit is granted if all of the following apply:\n\n"
        "● The holder owns a vehicle.\n\n"
        "● Any of the following apply:\n\n"
        "● The holder lives in the district.\n\n"
        "● The holder works in the district.\n\n"
        "● The holder has paid the fee."
    )
    (only,) = structure.parse_source_lists(text)
    assert (only.reading, only.binds_items) == ("unrecognised", False)
    assert len(only.items) == 5


def test_a_plain_paragraph_ends_the_list_and_a_wrapped_line_does_not():
    text = (
        ANY_CHAPEAU
        + "\n\n● New vehicle\nwrapped words of the item\n\n● Lost card"
        + "\n\nNext comes a plain paragraph."
    )
    (only,) = structure.parse_source_lists(text)
    assert [text[item.start : item.end] for item in only.items] == [
        "● New vehicle\nwrapped words of the item",
        "● Lost card",
    ]


def test_a_marked_item_at_the_start_is_not_a_list_head():
    text = "● Any of the following apply:\n\n● New vehicle\n\n● Lost card"
    assert structure.parse_source_lists(text) == ()


def test_the_flat_reader_still_needs_two_facts():
    assert sc._source_conjunctive_fact_gates("Granted if the holder owns a car.") == ()


# --- (b) a list of alternatives is verified on the value of the formula ------------


def test_a_faithful_or_passes_with_the_switch_on_and_is_refused_on_the_flat_reading(
    monkeypatch,
):
    rules = [("renewal", FAITHFUL, [ANY_CHAPEAU])]
    assert _both(ANY_INPUTS, rules, ANY, monkeypatch) == (1, 0)


WRONG_ENCODINGS = {
    "and for or": (f"{A} and {B} and {C} and {D}", ANY_INPUTS),
    "one or turned into and": (f"({A} and {B}) or {C} or {D}", ANY_INPUTS),
    "dropped alternative": (f"{A} or {B} or {C}", ANY_INPUTS),
    "negated alternative": (f"{A} or {B} or {C} or not {D}", ANY_INPUTS),
    "negated rule": (f"not ({A} or {B} or {C} or {D})", ANY_INPUTS),
    "constant true": ("true", ANY_INPUTS),
    "constant false": ("false", ANY_INPUTS),
    "alternative named in a dead arm": (
        f"{A} or {B} or {C} or ({D} and false)",
        ANY_INPUTS,
    ),
    "alternative named next to a constant": (
        f"{A} or {B} or {C} or ({D} or true)",
        ANY_INPUTS,
    ),
    "alternative named in a contradiction": (
        f"{A} or {B} or {C} or ({D} and not {D})",
        ANY_INPUTS,
    ),
    "alternative only together with another": (
        f"{A} or {B} or {C} or ({D} and {A})",
        ANY_INPUTS,
    ),
    "one input for every alternative": (
        "renewal_event_occurred",
        ["renewal_event_occurred"],
    ),
    "one input for two alternatives": (
        f"{A} or {B} or change_of_employer_or_fee_amount_disputed_reported",
        [A, B, "change_of_employer_or_fee_amount_disputed_reported"],
    ),
    "invented extra condition": (
        f"({A} or {B} or {C} or {D}) and holder_is_senior",
        [*ANY_INPUTS, "holder_is_senior"],
    ),
    "invented negative condition": (
        f"({A} or {B} or {C} or {D}) and not holder_is_senior",
        [*ANY_INPUTS, "holder_is_senior"],
    ),
    "invented condition on one alternative": (
        f"{A} or {B} or {C} or ({D} and holder_is_senior)",
        [*ANY_INPUTS, "holder_is_senior"],
    ),
    "invented extra alternative": (
        f"{A} or {B} or {C} or {D} or holder_is_senior",
        [*ANY_INPUTS, "holder_is_senior"],
    ),
}


@pytest.mark.parametrize("name", sorted(WRONG_ENCODINGS))
@pytest.mark.parametrize("proof", ["chapeau", "chapeau and items"])
def test_a_wrong_encoding_of_the_alternatives_is_refused(name, proof, monkeypatch):
    formula, inputs = WRONG_ENCODINGS[name]
    excerpts = [ANY_CHAPEAU] if proof == "chapeau" else [ANY_CHAPEAU, *ANY_ITEMS]
    content = _candidate(inputs, [("renewal", formula, excerpts)])
    assert len(_findings(content, ANY, on=True, monkeypatch=monkeypatch)) == 1


def test_the_wrong_encodings_that_the_flat_reading_accepts(monkeypatch):
    """The flat reading counts inputs on a path; it does not read the OR."""

    accepted = sorted(
        name
        for name, (formula, inputs) in WRONG_ENCODINGS.items()
        if not _findings(
            _candidate(inputs, [("renewal", formula, [ANY_CHAPEAU])]),
            ANY,
            on=False,
            monkeypatch=monkeypatch,
        )
    )
    assert accepted == [
        "and for or",
        "constant false",
        "negated rule",
    ]


def test_a_wrong_dependency_does_not_stand_for_an_alternative(monkeypatch):
    rules = [
        ("fee_dispute", "false", ["When the fee amount is disputed"]),
        ("renewal", f"{A} or {B} or {C} or fee_dispute", [ANY_CHAPEAU]),
    ]
    assert _both(ANY_INPUTS, rules, ANY, monkeypatch)[1] == 1


def test_alternatives_through_derived_rules_pass(monkeypatch):
    rules = [
        ("fee_dispute", D, ["When the fee amount is disputed"]),
        ("employer_change", C, ["When a change of employer is reported"]),
        ("renewal", f"{A} or {B} or employer_change or fee_dispute", [ANY_CHAPEAU]),
    ]
    assert _both(ANY_INPUTS, rules, ANY, monkeypatch)[1] == 0


@pytest.mark.parametrize(
    ("formula", "dtype"),
    [
        (f"if {FAITHFUL}: 1 else: 0", "Integer"),
        (FAITHFUL, "Money"),
        (f"({A} or {B} or {C} or {D}) == True", "Judgment"),
        (f"count_true({A}, {B}, {C}, {D}) >= 1", "Judgment"),
    ],
)
def test_a_formula_outside_the_closed_boolean_reader_keeps_the_flat_reading(
    formula, dtype, monkeypatch
):
    """Also a Boolean formula on a rule that is not declared Boolean."""

    off, on = _both(
        ANY_INPUTS, [("renewal", formula, [ANY_CHAPEAU])], ANY, monkeypatch, dtype
    )
    assert on == off


def test_an_item_without_a_readable_fact_keeps_the_flat_reading(monkeypatch):
    source = ANY + "\n\n● It"
    rules = [("renewal", FAITHFUL, [ANY_CHAPEAU])]
    assert _both(ANY_INPUTS, rules, source, monkeypatch) == (1, 1)


def test_an_excerpt_at_two_places_keeps_the_flat_reading(monkeypatch):
    """Ambiguous ownership is refused as before, not verified per clause."""

    source = ANY + "\n\nRepeated for the second office.\n\n" + ANY
    rules = [("renewal", FAITHFUL, [ANY_CHAPEAU])]
    off, on = _both(ANY_INPUTS, rules, source, monkeypatch)
    assert off == on == 1


def test_a_list_with_unmarked_children_keeps_the_flat_reading_for_its_own_rule(
    monkeypatch,
):
    inputs = [
        "new_permit_requested",
        "renewal_change_of_more_than_one_vehicle",
        "fee_amount_questionable",
    ]
    rules = [("proof_needed", " or ".join(inputs), [NESTED.split("\n")[0]])]
    off, on = _both(inputs, rules, NESTED, monkeypatch)
    assert off == on


CONJUNCT_ITEM = (
    "A parking fee is waived when one of the following applies:\n\n"
    "● They live in a campus residence, and their rent is zero\n\n"
    "● They hold a staff permit"
)
CONJUNCT_INPUTS = ["lives_in_campus_residence", "rent_is_zero", "holds_staff_permit"]


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("(lives_in_campus_residence and rent_is_zero) or holds_staff_permit", 0),
        ("lives_in_campus_residence or holds_staff_permit", 1),
        ("rent_is_zero or holds_staff_permit", 1),
        ("lives_in_campus_residence or rent_is_zero or holds_staff_permit", 1),
        ("lives_in_campus_residence and rent_is_zero and holds_staff_permit", 1),
        ("lives_in_campus_residence and rent_is_zero", 1),
    ],
)
def test_a_conjunction_inside_one_alternative_is_demanded(
    formula, expected, monkeypatch
):
    chapeau = CONJUNCT_ITEM.split("\n")[0]
    rules = [("waiver", formula, [chapeau])]
    assert _both(CONJUNCT_INPUTS, rules, CONJUNCT_ITEM, monkeypatch)[1] == expected


CHAPEAU_FACT = (
    "When the holder is enrolled in the commuter program, a parking fee is "
    "waived when one or more of the following apply:\n\n"
    "● The holder lives outside the district\n\n"
    "● The holder works a night shift"
)
FACT_INPUTS = [
    "holder_enrolled_in_commuter_program",
    "holder_lives_outside_district",
    "holder_works_night_shift",
]


@pytest.mark.parametrize(
    "formula",
    [
        "holder_enrolled_in_commuter_program and "
        "(holder_lives_outside_district or holder_works_night_shift)",
        "holder_lives_outside_district or holder_works_night_shift",
        "holder_enrolled_in_commuter_program or holder_lives_outside_district "
        "or holder_works_night_shift",
        "holder_enrolled_in_commuter_program",
    ],
)
def test_a_chapeau_with_a_second_condition_keeps_the_flat_reading(formula, monkeypatch):
    chapeau = CHAPEAU_FACT.split("\n")[0]
    (only,) = structure.parse_source_lists(CHAPEAU_FACT)
    assert only.reading == "alternatives" and not only.verifiable
    content = _candidate(FACT_INPUTS, [("waiver", formula, [chapeau])])
    assert _findings(
        content, CHAPEAU_FACT, on=True, monkeypatch=monkeypatch
    ) == _findings(content, CHAPEAU_FACT, on=False, monkeypatch=monkeypatch)


# --- clause bounds on the tree -----------------------------------------------------

TAIL = "The office keeps a copy of each permit."


def test_text_after_the_list_keeps_the_flat_reading(monkeypatch):
    """No closed form tells that a later paragraph adds nothing to the list."""

    source = ANY + "\n\n" + TAIL
    (only,) = structure.parse_source_lists(source)
    assert only.reading == "alternatives" and not only.verifiable
    rules = [
        ("renewal", FAITHFUL, [ANY_CHAPEAU]),
        ("copy_kept", "office_keeps_copy", [TAIL]),
    ]
    assert _both([*ANY_INPUTS, "office_keeps_copy"], rules, source, monkeypatch) == (
        1,
        1,
    )


@pytest.mark.parametrize(
    "tail",
    [
        "and the holder has no unpaid fines.",
        "provided that the holder has no unpaid fines.",
        "unpaid fines are checked when the holder has a second vehicle.",
    ],
)
def test_a_tail_that_starts_no_new_sentence_keeps_the_flat_reading(tail, monkeypatch):
    source = ANY + "\n\n" + tail
    off, on = _both(
        ANY_INPUTS, [("renewal", FAITHFUL, [ANY_CHAPEAU])], source, monkeypatch
    )
    assert (off, on) == (1, 1)


def test_an_excerpt_across_the_end_of_the_list_keeps_the_flat_reading(monkeypatch):
    source = ANY + "\n\n" + TAIL
    excerpt = "When the fee amount is disputed The office keeps a copy"
    off, on = _both(
        ANY_INPUTS, [("renewal", FAITHFUL, [ANY_CHAPEAU, excerpt])], source, monkeypatch
    )
    assert (off, on) == (1, 1)


def test_a_cue_before_the_chapeau_keeps_the_flat_reading():
    """The list's own rule is never read without its items."""

    first = "A fee is refunded if the holder is a resident and"
    source = (
        first
        + "\n\nAny of the following proofs is on file:\n\n"
        + "● A lease that shows the address.\n\n"
        + "● A bill that states the address."
    )
    lists = structure.parse_source_lists(source)
    assert [found.reading for found in lists] == ["alternatives"]
    clause = sc._SourceConditionClause((), 0, len(source), source)
    for excerpt in (first, "Any of the following proofs is on file:", source):
        assert (
            sc._structural_clause_reading(
                clause,
                excerpt,
                source_text=source,
                source_lists=lists,
                care_source_text="",
            )
            is None
        )


NOTE_TAIL = (
    ":\n\n● Fees charged to a company account\n\n"
    "NOTE When no due date is shown for a fee, the fee is considered due in the "
    "next month.\n\n"
    "● Late fees when the holder is paid monthly\n\n"
    "● Fees that were paid before a new permit month"
)
NOTE_QUOTE = (
    "When no due date is shown for a fee, the fee is considered due in the next month."
)


def test_a_note_in_a_list_of_alternatives_does_not_run_on_into_the_items(monkeypatch):
    source = "Any of the following parking fees are not refundable" + NOTE_TAIL
    rules = [("due", "fee_has_no_due_date_shown", [NOTE_QUOTE])]
    assert _both(["fee_has_no_due_date_shown"], rules, source, monkeypatch) == (1, 0)


def test_a_note_without_a_full_stop_keeps_the_flat_reading(monkeypatch):
    source = "Any of the following parking fees are not refundable" + NOTE_TAIL
    source = source.replace("next month.", "next month and")
    quote = NOTE_QUOTE.replace("next month.", "next month and")
    rules = [("due", "fee_has_no_due_date_shown", [quote])]
    off, on = _both(["fee_has_no_due_date_shown"], rules, source, monkeypatch)
    assert off == on == 1


@pytest.mark.parametrize(
    "excerpt",
    [
        # The proof runs on into the next item, or stands before the list.
        "the fee is considered due in the next month. ● Late fees when the "
        "holder is paid monthly",
        "Fees are listed below",
    ],
)
def test_a_proof_outside_the_note_keeps_the_flat_reading(excerpt, monkeypatch):
    source = (
        "Fees are listed below\n\n"
        "Any of the following parking fees are not refundable" + NOTE_TAIL
    )
    rules = [("due", "fee_has_no_due_date_shown", [excerpt])]
    off, on = _both(["fee_has_no_due_date_shown"], rules, source, monkeypatch)
    assert off == on


@pytest.mark.parametrize(
    ("formula", "inputs", "expected"),
    [
        ("fee_has_no_due_date_shown", ["fee_has_no_due_date_shown"], 0),
        (
            "fee_has_no_due_date_shown and fee_is_late",
            ["fee_has_no_due_date_shown", "fee_is_late"],
            0,
        ),
        ("true", ["fee_has_no_due_date_shown"], 1),
        ("not fee_has_no_due_date_shown", ["fee_has_no_due_date_shown"], 1),
        ("holder_is_senior", ["holder_is_senior"], 1),
        (
            "fee_has_no_due_date_shown or holder_is_senior",
            ["fee_has_no_due_date_shown", "holder_is_senior"],
            1,
        ),
    ],
)
def test_the_one_fact_of_a_cut_note_is_still_demanded(
    formula, inputs, expected, monkeypatch
):
    """The cut removes the run-on; the note's own fact needs a positive input."""

    in_list = "Any of the following parking fees are not refundable" + NOTE_TAIL
    rules = [("due", formula, [NOTE_QUOTE])]
    assert _both(inputs, rules, in_list, monkeypatch)[1] == expected


PROOF_CHAPEAU = "Any of the following can be used to prove residence:"
PROOF_ITEMS = (
    "A signed lease.",
    "A landlord letter when the letter states the monthly rent.",
    "A utility bill that shows the service address.",
    "A bank statement when the holder is billed by a relative.",
)
PROOF_INPUTS = [
    "signed_lease_provided",
    "landlord_letter_states_monthly_rent",
    "utility_bill_shows_service_address",
    "bank_statement_provided_and_holder_billed_by_relative",
]


@pytest.mark.parametrize(
    "formula",
    [
        " or ".join(PROOF_INPUTS),
        " or ".join(PROOF_INPUTS[:3]),
        " and ".join(PROOF_INPUTS),
        "true",
    ],
)
def test_a_rule_that_quotes_the_chapeau_is_never_cut(formula, monkeypatch):
    """Items with their own conditions are not plain alternatives: flat."""

    source = _list(PROOF_CHAPEAU, PROOF_ITEMS)
    content = _candidate(
        PROOF_INPUTS, [("proved", formula, [PROOF_CHAPEAU, *PROOF_ITEMS])]
    )
    assert _findings(content, source, on=True, monkeypatch=monkeypatch) == _findings(
        content, source, on=False, monkeypatch=monkeypatch
    )


@pytest.mark.parametrize(
    "chapeau",
    [
        "The following parking fees are not refundable",
        "All of the following parking fees are not refundable",
        "The holder must meet the following",
        "Fees are refunded if the holder meets the following",
    ],
)
def test_a_note_in_any_other_list_keeps_the_flat_reading(chapeau, monkeypatch):
    rules = [("due", "fee_has_no_due_date_shown", [NOTE_QUOTE])]
    assert _both(
        ["fee_has_no_due_date_shown"], rules, chapeau + NOTE_TAIL, monkeypatch
    ) == (1, 1)


NESTED = (
    "Proof is needed when any of the following occur:\n\n"
    "● A new permit is requested and when any of the following occur:\n\n"
    "When the holder is new to the district, proof of address must be shown.\n\n"
    "When the holder has unpaid fines that exceed the limit, all fines must be "
    "verified.\n\n"
    "● For renewals when a change of more than one vehicle must be verified.\n\n"
    "● The fee amount is questionable."
)
NESTED_QUOTE = (
    "When the holder has unpaid fines that exceed the limit, all fines must be "
    "verified."
)


def test_a_sentence_under_one_item_does_not_run_on_into_the_sibling_items(monkeypatch):
    rules = [("all_fines", "holder_has_unpaid_fines_exceeding_limit", [NESTED_QUOTE])]
    assert _both(
        ["holder_has_unpaid_fines_exceeding_limit"], rules, NESTED, monkeypatch
    ) == (1, 0)


def test_the_same_sentence_under_a_conjunctive_list_keeps_the_flat_reading(monkeypatch):
    source = NESTED.replace("when any of the following occur:", "when all occur:", 1)
    rules = [("all_fines", "holder_has_unpaid_fines_exceeding_limit", [NESTED_QUOTE])]
    assert _both(
        ["holder_has_unpaid_fines_exceeding_limit"], rules, source, monkeypatch
    ) == (1, 1)


# --- (c) a rule that quotes only items is read against the whole list --------------


COUNT_CHAPEAU = (
    "A parking permit is granted if the holder meets at least 2 of the "
    "following conditions:"
)
BINDING_CHAPEAUS = [
    ALL_CHAPEAU,
    "A parking permit is granted when each of the following is true:",
    "A parking permit is granted only if all of the following conditions are met:",
    "A parking permit is granted when the holder satisfies all of the following:",
    "A parking permit is granted when all the following apply:",
    "A parking permit is granted if both of the following apply:",
    "A parking permit is granted if every one of the following applies:",
    "A parking permit is granted provided that all of the following apply:",
    "A parking permit is granted if all three of the following apply:",
    "A parking permit is granted if the following conditions are met:",
    "A parking permit is granted when the following requirements are met:",
    "A parking permit is granted if:",
]
# The items of these lists are not put under the chapeau's cue by a closed
# form: a consequence list, a count, a cue far from the quantifier, a
# negation, or no cue at all.
NON_BINDING_CHAPEAUS = [
    PLAIN_CHAPEAU,
    COUNT_CHAPEAU,
    "A parking permit is granted if it meets two or more of the following criteria:",
    "When the holder moves, in the first month, all of the following apply:",
    "If the holder moves and the fee is unpaid then all of the following apply:",
    "A parking permit is granted if not all of the following apply:",
    "A parking permit is granted if the holder who owns a second large vehicle "
    "meets all of the following:",
    "A parking permit is granted if the holder fails all of the following:",
    "A parking permit is granted unless all of the following apply:",
    "The holder must provide all of the following:",
    "All of the following must be met:",
    "A parking permit is granted if all of the following fail:",
    "A parking permit is granted if the holder meets the following:",
    "A parking permit is granted if all of the following apply to nobody:",
]


@pytest.mark.parametrize("chapeau", BINDING_CHAPEAUS)
def test_the_closed_condition_list_forms_bind_item_proofs(chapeau):
    (only,) = structure.parse_source_lists(_list(chapeau, ALL_ITEMS))
    assert only.binds_items and only.reading != "alternatives"


@pytest.mark.parametrize("chapeau", NON_BINDING_CHAPEAUS)
def test_any_other_chapeau_binds_no_item_proof(chapeau, monkeypatch):
    (only,) = structure.parse_source_lists(_list(chapeau, ALL_ITEMS))
    assert not only.binds_items
    rules = [("granted", ALL_INPUTS[0], [ALL_ITEMS[0]])]
    assert _both(ALL_INPUTS, rules, _list(chapeau, ALL_ITEMS), monkeypatch) == (0, 0)


def test_an_unreadable_list_binds_no_item_proof():
    text = ALL_CHAPEAU + "\n\n● The holder owns a vehicle:\n\n● A car\n\n● A van"
    (only,) = structure.parse_source_lists(text)
    assert (only.reading, only.binds_items) == ("unrecognised", False)


@pytest.mark.parametrize("chapeau", BINDING_CHAPEAUS)
def test_a_dropped_conjunct_is_refused_when_only_items_are_quoted(chapeau, monkeypatch):
    source = _list(chapeau, ALL_ITEMS)
    rules = [("granted", " and ".join(ALL_INPUTS[:2]), list(ALL_ITEMS[:2]))]
    # The flat reading sees no list without a chapeau excerpt: 0 findings.
    assert _both(ALL_INPUTS, rules, source, monkeypatch) == (0, 1)


def test_item_only_proof_of_a_complete_conjunction_passes(monkeypatch):
    source = _list(ALL_CHAPEAU, ALL_ITEMS)
    rules = [("granted", " and ".join(ALL_INPUTS), list(ALL_ITEMS))]
    assert _both(ALL_INPUTS, rules, source, monkeypatch) == (0, 0)


def test_one_rule_for_each_item_passes_when_a_rule_combines_them(monkeypatch):
    source = _list(ALL_CHAPEAU, ALL_ITEMS) + "\n\nThe clerk then issues the permit."
    rules = [
        ("owns", ALL_INPUTS[0], [ALL_ITEMS[0]]),
        ("lives", ALL_INPUTS[1], [ALL_ITEMS[1]]),
        ("paid", ALL_INPUTS[2], [ALL_ITEMS[2]]),
        ("granted", "owns and lives and paid", ["The clerk then issues the permit."]),
    ]
    assert _both(ALL_INPUTS, rules, source, monkeypatch) == (0, 0)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_one_rule_for_each_item_without_a_combining_rule_is_not_bound(
    count, monkeypatch
):
    """Such rules say nothing about the whole list: main's result stays."""

    source = _list(ALL_CHAPEAU, ALL_ITEMS)
    rules = [
        (f"rule_{index}", ALL_INPUTS[index], [ALL_ITEMS[index]])
        for index in range(count)
    ]
    assert _both(ALL_INPUTS, rules, source, monkeypatch) == (0, 0)


def test_a_combining_rule_that_drops_an_item_rule_is_refused(monkeypatch):
    source = _list(ALL_CHAPEAU, ALL_ITEMS) + "\n\nThe clerk then issues the permit."
    rules = [
        ("owns", ALL_INPUTS[0], [ALL_ITEMS[0]]),
        ("lives", ALL_INPUTS[1], [ALL_ITEMS[1]]),
        ("paid", ALL_INPUTS[2], [ALL_ITEMS[2]]),
        ("granted", "owns and lives", ["The clerk then issues the permit."]),
    ]
    assert _both(ALL_INPUTS, rules, source, monkeypatch) == (0, 1)


def test_a_count_list_is_not_bound(monkeypatch):
    source = _list(COUNT_CHAPEAU, ALL_ITEMS)
    a, b, c = ALL_INPUTS
    rules = [
        ("pair_one", f"{a} and {b}", list(ALL_ITEMS[:2])),
        ("pair_two", f"{a} and {c}", [ALL_ITEMS[0], ALL_ITEMS[2]]),
        ("pair_three", f"{b} and {c}", list(ALL_ITEMS[1:])),
    ]
    assert _both(ALL_INPUTS, rules, source, monkeypatch) == (0, 0)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [(FAITHFUL, 0), (f"{A} or {B} or {C}", 1), (f"{A} and {B} and {C} and {D}", 1)],
)
def test_item_only_proof_of_a_list_of_alternatives_is_verified(
    formula, expected, monkeypatch
):
    rules = [("renewal", formula, list(ANY_ITEMS))]
    assert _both(ANY_INPUTS, rules, ANY, monkeypatch) == (0, expected)


def test_a_note_inside_a_list_binds_nothing(monkeypatch):
    source = (
        ALL_CHAPEAU
        + "\n\n● The holder owns a vehicle.\n\n"
        + "● The holder has paid the fee.\n\n"
        + "NOTE The clerk records the plate number of the vehicle.\n\n"
        + "● The holder lives in the district."
    )
    rules = [("plate", "plate_number_recorded", ["The clerk records the plate number"])]
    assert _both(["plate_number_recorded"], rules, source, monkeypatch) == (0, 0)


# --- (d) a conjunctive list is protected at least as on the flat reading -----------

CONJUNCTIVE_FORMULAS = {
    "faithful": (" and ".join(ALL_INPUTS), ALL_INPUTS),
    "dropped": (" and ".join(ALL_INPUTS[:2]), ALL_INPUTS),
    "merged": ("holder_qualifies", ["holder_qualifies"]),
    "two merged": (
        "holder_owns_vehicle and holder_lives_in_district_and_has_paid",
        ["holder_owns_vehicle", "holder_lives_in_district_and_has_paid"],
    ),
    "negated": (
        "holder_owns_vehicle and holder_lives_in_district and not holder_has_paid_fee",
        ALL_INPUTS,
    ),
    "or for and": (" or ".join(ALL_INPUTS), ALL_INPUTS),
    "constant": ("true", ALL_INPUTS),
    "constant folded": (
        "holder_owns_vehicle and holder_lives_in_district and (holder_has_paid_fee or true)",
        ALL_INPUTS,
    ),
}


@pytest.mark.parametrize("name", sorted(CONJUNCTIVE_FORMULAS))
@pytest.mark.parametrize("chapeau", [ALL_CHAPEAU, PLAIN_CHAPEAU])
@pytest.mark.parametrize("proof", ["chapeau", "chapeau and items", "items"])
def test_a_conjunctive_list_is_refused_at_least_as_often_as_on_the_flat_reading(
    name, chapeau, proof, monkeypatch
):
    formula, inputs = CONJUNCTIVE_FORMULAS[name]
    excerpts = {
        "chapeau": [chapeau],
        "chapeau and items": [chapeau, *ALL_ITEMS],
        "items": list(ALL_ITEMS),
    }[proof]
    content = _candidate(inputs, [("granted", formula, excerpts)])
    source = _list(chapeau, ALL_ITEMS)
    off = _findings(content, source, on=False, monkeypatch=monkeypatch)
    on = _findings(content, source, on=True, monkeypatch=monkeypatch)
    if proof == "items":
        assert len(on) >= len(off)
    else:
        # The chapeau is quoted: the two readings give the same finding text.
        assert on == off


# --- (a) fail closed ---------------------------------------------------------------

_WORDS = st.sampled_from(
    "holder permit fee vehicle district owns lives pays has is all each every "
    "both none not no never unless when if and or the following these conditions "
    "apply occur met must any one either of two together only".split()
)
_CHAPEAUS = st.lists(_WORDS, min_size=2, max_size=12).map(
    lambda words: " ".join(words).capitalize() + ":"
)


@settings(max_examples=300, deadline=None)
@given(
    chapeau=_CHAPEAUS,
    items=st.lists(st.sampled_from(ALL_ITEMS), min_size=2, max_size=3),
)
def test_a_text_without_a_closed_chapeau_form_is_read_exactly_as_flat(chapeau, items):
    """Without the closed evidence the structural reading declines every clause."""

    collapsed = " ".join(chapeau.split())
    closed = (
        structure._ANY_CUE_CHAPEAU.search(collapsed) is not None
        or structure._ANY_SUBJECT_CHAPEAU.fullmatch(collapsed) is not None
    ) and len(structure.re.findall(r"\bfollowing\b", collapsed, structure.re.I)) == 1
    source = _list(chapeau, items)
    lists = structure.parse_source_lists(source)
    assert any(found.reading == "alternatives" for found in lists) == closed
    if closed:
        return
    clause = sc._SourceConditionClause((), 0, len(source), source)
    for excerpt in (chapeau, items[0], source):
        assert (
            sc._structural_clause_reading(
                clause,
                excerpt,
                source_text=source,
                source_lists=lists,
                care_source_text="",
            )
            is None
        )


@pytest.mark.parametrize("chapeau", NOT_ALTERNATIVES)
@pytest.mark.parametrize("formula", [FAITHFUL, f"{A} and {B} and {C} and {D}", A])
def test_a_chapeau_outside_the_closed_forms_gives_the_flat_result(
    chapeau, formula, monkeypatch
):
    source = _list(chapeau, ANY_ITEMS)
    content = _candidate(ANY_INPUTS, [("renewal", formula, [chapeau])])
    assert _findings(content, source, on=True, monkeypatch=monkeypatch) == _findings(
        content, source, on=False, monkeypatch=monkeypatch
    )


def test_a_unit_with_a_reviewed_reading_is_not_read_structurally(monkeypatch):
    """The registry stays the stronger exemption: such a unit is read as before."""

    monkeypatch.setattr(sc, "reviewed_alternative_lists_for", lambda *_args: ("entry",))
    monkeypatch.setattr(
        sc, "_clause_is_reviewed_alternative_list", lambda *_args: False
    )

    def unreachable(*_args, **_kwargs):
        raise AssertionError("structural reading ran on a reviewed unit")

    monkeypatch.setattr(structure, "parse_source_lists", unreachable)
    content = _candidate(ANY_INPUTS, [("renewal", FAITHFUL, [ANY_CHAPEAU])])
    assert len(_findings(content, ANY, on=True, monkeypatch=monkeypatch)) == 1


def test_an_excerpt_that_stands_at_two_places_binds_nothing(monkeypatch):
    source = _list(ALL_CHAPEAU, ALL_ITEMS) + "\n\nRecord: The holder owns a vehicle."
    rules = [("granted", ALL_INPUTS[0], [ALL_ITEMS[0]])]
    assert _both(ALL_INPUTS, rules, source, monkeypatch) == (0, 0)


def test_every_pair_of_wrong_alternative_encodings_is_distinct():
    formulas = [formula for formula, _inputs in WRONG_ENCODINGS.values()]
    assert all(left != right for left, right in itertools.combinations(formulas, 2))


# --- review round 1: forms that must keep the flat reading ---------------------------

SHORT_ITEMS = (
    "The holder owns a vehicle",
    "When a change of employer is reported",
    "When the fee amount is disputed",
)
SHORT_INPUTS = [
    "holder_owns_vehicle",
    "change_of_employer_reported",
    "fee_amount_disputed",
]
SHORT_OR = " or ".join(SHORT_INPUTS)


def _same(inputs, rules, source, monkeypatch, dtype="Judgment"):
    """Assert that the switch changes no finding; return the count."""

    content = _candidate(inputs, rules, dtype)
    off = _findings(content, source, on=False, monkeypatch=monkeypatch)
    assert _findings(content, source, on=True, monkeypatch=monkeypatch) == off
    return len(off)


def test_the_short_list_is_verified(monkeypatch):
    source = _list(ANY_CHAPEAU, SHORT_ITEMS)
    rules = [("renewal", SHORT_OR, [ANY_CHAPEAU])]
    assert _both(SHORT_INPUTS, rules, source, monkeypatch) == (1, 0)


@pytest.mark.parametrize(
    "middle",
    [
        "Page 12 of 30",
        "(continued)",
        "Parking Manual, Section 4",
        "Revised 01/2026",
        "12",
        "Renewal of permits:",
        "The list continues on the next page.",
    ],
)
@pytest.mark.parametrize("keep", [1, 2])
@pytest.mark.parametrize("formula", ["first", SHORT_OR])
def test_an_interrupted_list_is_not_read_at_all(middle, keep, formula, monkeypatch):
    source = (
        ANY_CHAPEAU
        + "".join(f"\n\n● {item}" for item in SHORT_ITEMS[:keep])
        + f"\n\n{middle}"
        + "".join(f"\n\n● {item}" for item in SHORT_ITEMS[keep:])
    )
    assert not any(found.verifiable for found in structure.parse_source_lists(source))
    formula = " or ".join(SHORT_INPUTS[:keep]) if formula == "first" else formula
    _same(SHORT_INPUTS, [("renewal", formula, [ANY_CHAPEAU])], source, monkeypatch)


def test_a_marked_item_outside_every_list_declines_the_whole_unit():
    source = _list(ANY_CHAPEAU, SHORT_ITEMS) + "\n\nA plain line.\n\n● A stray item"
    assert structure.parse_source_lists(source) == ()


@pytest.mark.parametrize(
    "formula",
    [
        f"({A} or {B} or {C} or {D}) and not ({A} and {B})",
        f"({A} or {B} or {C} or {D}) and not ({A} and {B} and {C} and {D})",
        f"{A} != {B} or {C} or {D}",
    ],
)
def test_a_formula_that_is_not_a_plain_or_is_refused(formula, monkeypatch):
    """An alternative must grant the rule whatever the others are."""

    rules = [("renewal", formula, [ANY_CHAPEAU])]
    assert _both(ANY_INPUTS, rules, ANY, monkeypatch)[1] == 1


def test_an_equivalent_spelling_of_the_or_passes(monkeypatch):
    formula = f"({A} and not {B} and not {C} and not {D}) or {B} or {C} or {D}"
    rules = [("renewal", formula, [ANY_CHAPEAU])]
    assert _both(ANY_INPUTS, rules, ANY, monkeypatch)[1] == 0


REVERSED_CHAPEAUS = [
    "A parking permit is valid except when any of the following occur:",
    "A parking permit is valid, but not if any of the following occur:",
    "A parking permit is not valid when any of the following occur:",
    "Do not renew a parking permit if any of the following occur:",
    "A parking permit is never renewed when any of the following occur:",
    "No parking permit is renewed when any of the following occur:",
    "A parking permit cannot be renewed if any of the following occur:",
    "A parking permit is denied when any of the following occur:",
    "A parking permit is invalid when any of the following occur:",
    "A parking permit is renewed without a fee when any of the following occur:",
    "A parking permit is renewed even if any of the following occur:",
    "A parking permit is renewed when each and if any of the following apply:",
    "A parking permit is renewed when the clerk asks if any of the following occur:",
    "Unless the holder objects, a permit is renewed when any of the following occur:",
    "A parking permit stops being valid when any of the following occur:",
    "A parking permit isn't renewed when any of the following occur:",
    "Renew a parking permit when any of the following occur:",
    "Any of the following can not be used to prove residence:",
]


@pytest.mark.parametrize("chapeau", REVERSED_CHAPEAUS)
@pytest.mark.parametrize(
    "formula", [SHORT_OR, f"not ({SHORT_OR})", "holder_owns_vehicle"]
)
def test_a_chapeau_that_is_not_a_plain_affirmative_keeps_the_flat_reading(
    chapeau, formula, monkeypatch
):
    source = _list(chapeau, SHORT_ITEMS)
    assert not any(found.verifiable for found in structure.parse_source_lists(source))
    _same(SHORT_INPUTS, [("valid", formula, [chapeau])], source, monkeypatch)


QUALIFIED_ITEMS = [
    "The holder does not own a vehicle",
    "The holder owns no vehicle",
    "The holder owns a vehicle, provided that the fee is paid",
    "The holder both owns a vehicle and pays the fee",
    "The holder owns a vehicle as well as a garage",
    "The holder owns a vehicle but only when the fee is paid",
    "The holder owns a vehicle if the fee is paid",
    "The holder owns a vehicle that is registered in the district",
    "The holder who lives in the district owns a vehicle",
    "The holder reports a loss with a police report",
    "The holder owns more than one vehicle",
    "The holder owns a vehicle or leases a vehicle or rents a vehicle",
    "The holder owns a vehicle and",
    "the holder owns a vehicle",
    "The holder owns a vehicle; the fee is paid",
    "The holder owns a vehicle: a car",
    "The holder owns a vehicle unless the fee is unpaid",
    "The holder owns a vehicle, the fee is paid",
]


@pytest.mark.parametrize("item", QUALIFIED_ITEMS)
@pytest.mark.parametrize(
    "formula",
    [SHORT_OR, f"(holder_owns_vehicle and fee_is_paid) or {SHORT_OR[20:]}"],
)
def test_an_item_that_is_not_one_plain_alternative_keeps_the_flat_reading(
    item, formula, monkeypatch
):
    assert not structure.item_is_plain(item)
    source = _list(ANY_CHAPEAU, (item, *SHORT_ITEMS[1:]))
    inputs = [*SHORT_INPUTS, "fee_is_paid"]
    _same(inputs, [("renewal", formula, [ANY_CHAPEAU])], source, monkeypatch)


@pytest.mark.parametrize(
    "item",
    [
        "New vehicle",
        "When a change of employer is reported",
        "If the fee amount is disputed (See Fees for more information)",
        "They live in a campus residence, and their rent is zero",
        "When the fee amount is questionable(g)",
        "A change in the expense or address is received",
    ],
)
def test_a_plain_item_is_accepted(item):
    assert structure.item_is_plain(item)


ADDING_TAILS = [
    "Provided that the holder lives in the district.",
    "Unless the holder has moved away.",
    "In addition, the holder must live in the district.",
    "The holder must also live in the district.",
    "The holder also lives in the district.",
    "And the holder lives in the district.",
    "and the holder lives in the district.",
    "Only holders who live in the district qualify.",
    "A proof of address is required.",
    "A proof of address is needed to renew.",
    "This applies if the holder lives in the district.",
    "The above applies to residents.",
    "In addition, all of the following must apply:\n\n● The holder is an adult\n\n● The holder is insured",
    "Renewal conditions:",
    "The holder lives in the district",
]


@pytest.mark.parametrize("tail", ADDING_TAILS)
@pytest.mark.parametrize("period", ["", "."])
@pytest.mark.parametrize(
    "formula", [SHORT_OR, f"({SHORT_OR}) and holder_lives_in_district"]
)
def test_a_paragraph_that_may_add_to_the_list_keeps_the_flat_reading(
    tail, period, formula, monkeypatch
):
    source = _list(ANY_CHAPEAU, [item + period for item in SHORT_ITEMS]) + "\n\n" + tail
    assert not any(found.verifiable for found in structure.parse_source_lists(source))
    inputs = [*SHORT_INPUTS, "holder_lives_in_district"]
    _same(inputs, [("renewal", formula, [ANY_CHAPEAU])], source, monkeypatch)


OR_ITEM = ("The holder owns a vehicle or leases a garage", *SHORT_ITEMS[1:])
OR_REST = SHORT_OR[len("holder_owns_vehicle or ") :]


@pytest.mark.parametrize(
    ("first", "inputs", "expected"),
    [
        (
            "holder_owns_vehicle or holder_leases_garage",
            ["holder_owns_vehicle", "holder_leases_garage"],
            0,
        ),
        (
            "holder_owns_vehicle_or_leases_garage",
            ["holder_owns_vehicle_or_leases_garage"],
            0,
        ),
        ("holder_owns_vehicle", ["holder_owns_vehicle"], 1),
        ("holder_leases_garage", ["holder_leases_garage"], 1),
    ],
)
def test_an_item_with_one_or_needs_inputs_that_name_both_cases(
    first, inputs, expected, monkeypatch
):
    source = _list(ANY_CHAPEAU, OR_ITEM)
    rules = [("renewal", f"{first} or {OR_REST}", [ANY_CHAPEAU])]
    assert (
        _both([*inputs, *SHORT_INPUTS[1:]], rules, source, monkeypatch)[1] == expected
    )


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("false", 1),
        (f"({SHORT_OR}) and false", 1),
        ("not true", 1),
        ("true", 1),
        (f"({SHORT_OR}) or true", 1),
        (f"({SHORT_OR}) and true", 0),
        (f"{SHORT_OR} or false", 0),
    ],
)
def test_rulespec_boolean_literals_are_read_as_constants(
    formula, expected, monkeypatch
):
    source = _list(ANY_CHAPEAU, SHORT_ITEMS)
    rules = [("renewal", formula, [ANY_CHAPEAU])]
    assert _both(SHORT_INPUTS, rules, source, monkeypatch)[1] == expected


def _with_later_version(later):
    document = yaml.safe_load(
        _candidate(
            SHORT_INPUTS,
            [
                (
                    "renewal",
                    f"{SHORT_INPUTS[0]} or {SHORT_INPUTS[1]} or dispute",
                    [ANY_CHAPEAU],
                ),
                ("dispute", SHORT_INPUTS[2], []),
            ],
        )
    )
    document["rules"][1]["versions"].append(
        {"effective_from": "2027-01-01", "formula": later}
    )
    return yaml.safe_dump(document, sort_keys=False)


@pytest.mark.parametrize(
    ("later", "expected"),
    [(SHORT_INPUTS[2], 0), ("false", 1), (SHORT_INPUTS[0], 1), ("true", 1)],
)
def test_an_alternative_must_hold_in_every_version_of_a_dependency(
    later, expected, monkeypatch
):
    source = _list(ANY_CHAPEAU, SHORT_ITEMS)
    content = _with_later_version(later)
    assert len(_findings(content, source, on=True, monkeypatch=monkeypatch)) == expected


@pytest.mark.parametrize("dtype", ["Integer", "Money", "String"])
def test_an_input_that_is_not_boolean_keeps_the_flat_reading(dtype, monkeypatch):
    source = _list(ANY_CHAPEAU, SHORT_ITEMS)
    odd = {
        "name": SHORT_INPUTS[2],
        "entity": "Household",
        "dtype": dtype,
        "period": "Month",
    }
    rules = [("renewal", SHORT_OR, [ANY_CHAPEAU])]
    assert _same([*SHORT_INPUTS[:2], odd], rules, source, monkeypatch) == 1


TWO_FACT_ITEM = (
    "The holder owns a vehicle, and the holder lives in the district",
    *SHORT_ITEMS[1:],
)


@pytest.mark.parametrize(
    ("first", "inputs", "expected"),
    [
        (
            "(holder_owns_vehicle and holder_lives_in_district)",
            ["holder_owns_vehicle", "holder_lives_in_district"],
            0,
        ),
        (
            "holder_owns_vehicle_and_lives_in_district",
            ["holder_owns_vehicle_and_lives_in_district"],
            1,
        ),
        ("holder_owns_vehicle", ["holder_owns_vehicle", "holder_lives_in_district"], 1),
    ],
)
def test_one_input_does_not_stand_for_two_facts_of_an_item(
    first, inputs, expected, monkeypatch
):
    source = _list(ANY_CHAPEAU, TWO_FACT_ITEM)
    rules = [("renewal", f"{first} or {OR_REST}", [ANY_CHAPEAU])]
    assert (
        _both([*inputs, *SHORT_INPUTS[1:]], rules, source, monkeypatch)[1] == expected
    )


def test_an_alternative_whose_inputs_contain_another_is_refused(monkeypatch):
    items = (
        "The holder owns a vehicle",
        "The holder owns a vehicle, and the holder lives in the district",
        "When the fee amount is disputed",
    )
    inputs = ["holder_owns_vehicle", "holder_lives_in_district", "fee_amount_disputed"]
    formula = (
        "holder_owns_vehicle or (holder_owns_vehicle and holder_lives_in_district) "
        "or fee_amount_disputed"
    )
    rules = [("renewal", formula, [ANY_CHAPEAU])]
    assert _both(inputs, rules, _list(ANY_CHAPEAU, items), monkeypatch)[1] == 1


@pytest.mark.parametrize(
    "decoy",
    [
        ("never", "false", "Judgment"),
        ("label", "holder_owns_vehicle", "Judgment"),
    ],
)
def test_a_decoy_rule_that_quotes_a_cue_form_chapeau_is_itself_verified(
    decoy, monkeypatch
):
    source = _list(ANY_CHAPEAU, SHORT_ITEMS)
    name, formula, _dtype = decoy
    rules = [
        ("renewal", " or ".join(SHORT_INPUTS[:2]), list(SHORT_ITEMS[:2])),
        (name, formula, [ANY_CHAPEAU]),
    ]
    assert _both(SHORT_INPUTS, rules, source, monkeypatch)[1] == 1


@pytest.mark.parametrize(
    "name",
    ["fee_amount_not_disputed", "fee_amount_disputed_not", "no_fee_amount_disputed"],
)
def test_an_input_whose_name_negates_does_not_stand_for_an_alternative(
    name, monkeypatch
):
    source = _list(ANY_CHAPEAU, SHORT_ITEMS)
    inputs = [*SHORT_INPUTS[:2], name]
    rules = [("renewal", " or ".join(inputs), [ANY_CHAPEAU])]
    assert _both(inputs, rules, source, monkeypatch)[1] == 1


@pytest.mark.parametrize(
    ("first", "inputs"),
    [
        ("holder_lives_in_district", ["holder_lives_in_district"]),
        (
            "(new_vehicle and holder_lives_in_district)",
            ["new_vehicle", "holder_lives_in_district"],
        ),
    ],
)
def test_a_joint_item_with_a_part_that_is_no_counted_fact_keeps_the_flat_reading(
    first, inputs, monkeypatch
):
    items = ("New vehicle, and the holder lives in the district", *SHORT_ITEMS[1:])
    rules = [("renewal", f"{first} or {OR_REST}", [ANY_CHAPEAU])]
    _same([*inputs, *SHORT_INPUTS[1:]], rules, _list(ANY_CHAPEAU, items), monkeypatch)


@pytest.mark.parametrize(
    "item",
    [
        "The holder owns a vehicle (unless it is leased)",
        "The holder owns a vehicle (for residents)",
    ],
)
def test_a_parenthetical_that_is_no_reference_stays_in_the_item(item):
    assert not structure.item_is_plain(item)


@pytest.mark.parametrize(
    "item",
    [
        "The holder owns a vehicle subject to a paid deposit",
        "The holder owns a vehicle at a garage in the district",
        "The holder owns a vehicle twice in one year",
    ],
)
@pytest.mark.parametrize(
    "formula",
    [SHORT_OR, f"(holder_owns_vehicle and deposit_is_paid) or {SHORT_OR[23:]}"],
)
def test_an_item_word_that_no_input_names_keeps_the_flat_reading(
    item, formula, monkeypatch
):
    """A word outside every input name can be a qualifier that was dropped."""

    source = _list(ANY_CHAPEAU, (item, *SHORT_ITEMS[1:]))
    inputs = [*SHORT_INPUTS, "deposit_is_paid"]
    assert (
        _same(inputs, [("renewal", formula, [ANY_CHAPEAU])], source, monkeypatch) == 1
    )


def test_inputs_that_name_every_item_word_are_verified(monkeypatch):
    item = "The holder owns a vehicle at a garage in the district"
    source = _list(ANY_CHAPEAU, (item, *SHORT_ITEMS[1:]))
    first = "holder_owns_vehicle_at_garage_in_district"
    rules = [("renewal", f"{first} or {SHORT_OR[23:]}", [ANY_CHAPEAU])]
    assert _both([first, *SHORT_INPUTS[1:]], rules, source, monkeypatch) == (1, 0)
