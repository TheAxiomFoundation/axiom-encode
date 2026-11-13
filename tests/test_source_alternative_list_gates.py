"""Reviewed lists of alternatives are not simultaneous facts for gates-collapsed.

The fixtures are real rejected candidates of three Arizona utility units; see
``tests/fixtures/az-utility-alternative-lists/README.md``. The exception is a
registry entry bound to the citation and the exact text; see
``docs/reviewed-alternative-lists.md``. There is no text grammar.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import reviewed_alternative_lists as registry
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _source_conjunctive_fact_gates,
    _without_reviewed_alternative_lines,
    analyze_complete_source_unit,
)

FIXTURES = Path(__file__).parent / "fixtures" / "az-utility-alternative-lists"
UNIT = "us-az/manual/des/faa5/na-utility-expenses-and-allowances"

LIFELINE = (
    "When the budgetary unit participates in the Lifeline Assistance Program, a "
    "telephone expense is allowable when one or more of the following apply:\n\n"
    "● The discount program does not pay the expense in full\n\n"
    "● Extra airtime is purchased after exhausting free minutes"
)
SHARED = (
    "The budgetary unit is entitled to a utility allowance based on utility "
    "expenses verified when either situation occurs:\n\n"
    "● More than one budgetary unit lives in the same residence and shares "
    "utility expenses.\n\n"
    "● One budgetary unit shares a housing unit with others who do not receive NA."
)
HEATING = (FIXTURES / "block-3-07b0428f" / "source.txt").read_text()
WOOD = (
    "The budgetary unit is eligible for SUA when all of the following occur:\n\n"
    "● The wood, which is purchased or billed separate from or itemized with rent "
    "or mortgage payments\n\n"
    "● The wood is used to produce heat."
)


def _collapsed(
    directory: str,
    block: int,
    content: str | None = None,
    source: str | None = None,
    citation_block: int | None = None,
):
    folder = FIXTURES / directory
    citation = f"{UNIT}/block-{citation_block or block}"
    profile = vp._numeric_profile_for_citation_path(citation)
    analysis = analyze_complete_source_unit(
        content or (folder / "candidate.yaml").read_text(),
        source or (folder / "source.txt").read_text(),
        corpus_citation_path=citation,
        test_cases=yaml.safe_load((folder / "candidate.test.yaml").read_text()),
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


SIX = [
    ("block-14-e9371ed3", 14),
    ("block-14-66be7798", 14),
    ("block-3-07b0428f", 3),
    ("block-3-9dd68cae", 3),
    ("block-4-170e8d74", 4),
    ("block-4-28d570f4", 4),
]


@pytest.mark.parametrize(("directory", "block"), SIX)
def test_candidates_that_encode_alternatives_as_alternatives_pass(directory, block):
    assert _collapsed(directory, block) == []


@pytest.mark.parametrize(("directory", "block"), SIX)
def test_a_candidate_fails_without_its_registry_entry(directory, block, monkeypatch):
    monkeypatch.setattr(
        "axiom_encode.harness.source_completeness.reviewed_alternative_lists_for",
        lambda citation, text: (),
    )
    assert len(_collapsed(directory, block)) == 1


@pytest.mark.parametrize(("directory", "block"), SIX)
@pytest.mark.parametrize("error", [ValueError("malformed"), OSError("unreadable")])
def test_a_candidate_fails_when_the_registry_cannot_be_read(
    directory, block, error, monkeypatch
):
    def broken():
        raise error

    monkeypatch.setattr(registry, "load_reviewed_alternative_lists", broken)
    assert len(_collapsed(directory, block)) == 1


@pytest.mark.parametrize(("directory", "block"), SIX)
def test_an_entry_does_not_apply_under_another_citation(directory, block, monkeypatch):
    source = (FIXTURES / directory / "source.txt").read_text()
    citation = f"{UNIT}/block-{block}"
    (entry,) = registry.reviewed_alternative_lists_for(citation, source)
    for other in (f"{UNIT}/block-16", citation + "/1", citation.upper(), "", None):
        assert registry.reviewed_alternative_lists_for(other, source) == ()
    # The same reviewed text, recorded for another unit, does not help this one.
    elsewhere = dataclasses.replace(entry, citation_path=f"{UNIT}/block-99")
    monkeypatch.setattr(
        registry, "load_reviewed_alternative_lists", lambda: (elsewhere,)
    )
    assert registry.reviewed_alternative_lists_for(elsewhere.citation_path, source)
    assert len(_collapsed(directory, block)) == 1


@pytest.mark.parametrize(("directory", "block"), SIX)
def test_an_entry_does_not_apply_to_a_changed_text(directory, block):
    source = (FIXTURES / directory / "source.txt").read_text()
    (entry,) = registry.reviewed_alternative_lists_for(f"{UNIT}/block-{block}", source)
    bullet = next(line for line in entry.lines if line.reading == "alternative")
    last = bullet.physical_end - 1
    assert source[last].isalpha() or source[last] == "."
    changes = [
        # One character of a recorded bullet.
        source[:last] + ("x" if source[last] != "x" else "y") + source[last + 1 :],
        # A new bullet in the list.
        source[: bullet.physical_end]
        + "\n\n● The parent is employed"
        + source[bullet.physical_end :],
    ]
    # One character outside the recorded lines: only the digest differs.
    if entry.chapeau.physical_start:
        changes.append("Tho" + source[3:])
    if entry.lines[-1].physical_end < len(source):
        changes.append(source[:-1] + "!")
    assert len(changes) == (2 if block == 3 else 3)
    for changed in changes:
        assert changed != source
        assert (
            registry.reviewed_alternative_lists_for(entry.citation_path, changed) == ()
        )
        assert len(_collapsed(directory, block, source=changed)) == 1, changed


def test_all_of_the_following_list_still_passes_with_separate_facts():
    assert _collapsed("control-block-16-858a1c04", 16) == []


def test_merged_all_of_the_following_facts_still_fail():
    folder = FIXTURES / "control-block-16-858a1c04"
    payload = yaml.safe_load((folder / "candidate.yaml").read_text())
    template = payload["inputs"][0]
    payload["inputs"].append(
        {
            "name": "wood_sua_conditions_met",
            "entity": template["entity"],
            "dtype": "Boolean",
            "period": template["period"],
        }
    )
    merged = [
        rule
        for rule in payload["rules"]
        if "same_wood_used_to_produce_heat" in rule["versions"][0]["formula"]
    ]
    assert [rule["name"] for rule in merged] == ["wood_expense_sua_condition"]
    merged[0]["versions"][0]["formula"] = "wood_sua_conditions_met"

    issues = _collapsed(
        "control-block-16-858a1c04", 16, yaml.safe_dump(payload, sort_keys=False)
    )

    assert len(issues) == 1
    assert "`wood_expense_sua_condition` versions[0]" in issues[0]
    assert "when all of the following occur:" in issues[0]
    assert issues[0].expected == "2 distinct gate inputs"
    assert issues[0].found.startswith("0 of 2 source gates")


def test_merged_shared_meter_conjunction_still_fails_and_names_its_clause():
    folder = FIXTURES / "block-4-28d570f4"
    payload = yaml.safe_load((folder / "candidate.yaml").read_text())
    template = payload["inputs"][0]
    payload["inputs"].append(
        {
            "name": "evaluated_unit_metered_status",
            "entity": template["entity"],
            "dtype": "Boolean",
            "period": template["period"],
        }
    )
    rule = payload["rules"][-1]
    assert rule["name"] == "separate_housing_shared_meter_utility_allowance_path"
    rule["versions"][0]["formula"] = (
        "evaluated_unit_utility_expense_verified and evaluated_unit_metered_status"
    )

    issues = _collapsed("block-4-28d570f4", 4, yaml.safe_dump(payload, sort_keys=False))

    assert len(issues) == 1
    message = issues[0]
    # Only the rule with the real conjunction; the alternative paths stay clean.
    assert message.count(" versions[") == 1
    assert "`separate_housing_shared_meter_utility_allowance_path` versions[0]" in (
        message
    )
    assert (
        '[source clause "When budgetary units that incur utility expenses live '
        "in separate housing units" in message
    )
    assert "states 2 facts by their source words: fact 1 (" in message
    assert "gives 1 of 2 facts its own input" in message
    assert "no input name matches the source words of fact 2]" in message
    assert message.expected == "2 distinct gate inputs"
    assert message.found == (
        "1 of 2 source gates with their own terminal input on the "
        "least-supported formula path"
    )


def _bound(clause: str, block: int, source: str | None = None):
    """Return the entries for the unit and the offset of ``clause`` in it."""

    source = (
        source or (FIXTURES / dict(BLOCK_FIXTURE)[block] / "source.txt").read_text()
    )
    entries = registry.reviewed_alternative_lists_for(f"{UNIT}/block-{block}", source)
    return entries, source.index(clause)


BLOCK_FIXTURE = [(block, directory) for directory, block in SIX[::2]]
CLAUSES = [(LIFELINE, 14), (SHARED, 4), (HEATING, 3)]


@pytest.mark.parametrize(("clause", "block"), CLAUSES)
def test_a_bound_entry_drops_the_alternative_lines(clause, block):
    entries, start = _bound(clause, block)
    assert len(entries) == 1
    # Without the registry entry and the offset, every bullet is a gate.
    plain = _source_conjunctive_fact_gates(clause)
    assert plain == _source_conjunctive_fact_gates(
        clause, reviewed_alternative_lists=entries
    )
    assert plain == _source_conjunctive_fact_gates(clause, text_start=start)
    bound = _source_conjunctive_fact_gates(
        clause, reviewed_alternative_lists=entries, text_start=start
    )
    assert len(bound) < 2
    assert bound != plain or block == 3


def _reread(clause: str, block: int, *, body=None, shift=0, start_shift=0, text=None):
    entries, start = _bound(clause, block)
    cue = clause.index(" when ") + len(" when") if block != 14 else len("When")
    text = clause if text is None else text
    original = text[cue:] if body is None else body
    result = _without_reviewed_alternative_lines(
        text, original, cue + shift, start + start_shift, entries
    )
    return original, result


def test_the_reading_keeps_conjuncts_and_drops_the_rest():
    chapeau, first, second = SHARED.split("\n\n")
    original, result = _reread(SHARED, 4)
    assert result == original.split("\n")[0] + "\n" + first
    assert " and " in first

    original, result = _reread(HEATING, 3)
    assert result == original.split("\n")[0]
    original, result = _reread(LIFELINE, 14)
    assert result == original.split("\n")[0]


@pytest.mark.parametrize(("clause", "block"), CLAUSES)
def test_any_mismatch_with_the_recorded_place_keeps_the_body(clause, block):
    original, result = _reread(clause, block)
    assert result != original
    # The clause is at another offset of the source text.
    for start_shift in (-1, 1, 7):
        original, result = _reread(clause, block, start_shift=start_shift)
        assert result == original
    # An earlier step cut or rewrote the body.
    original, result = _reread(clause, block, body=clause[clause.index("●") :])
    assert result == original
    original, result = _reread(clause, block, shift=1)
    assert result == original
    # A line that is not recorded follows the list.
    extra = clause + "\n\nThe parent is employed and the child is disabled."
    original, result = _reread(clause, block, text=extra)
    assert result == original
    # A recorded line is missing.
    shorter = clause[: clause.rindex("●")].rstrip()
    original, result = _reread(clause, block, text=shorter)
    assert result == original
    # A recorded line has one other character, at the same offsets.
    last = clause.rindex("●") + 2
    changed = clause[:last] + ("x" if clause[last] != "x" else "y") + clause[last + 1 :]
    original, result = _reread(clause, block, text=changed)
    assert result == original
    # The chapeau has one other character.
    changed = clause.replace("when", "When", 1) if block != 14 else "w" + clause[1:]
    assert changed != clause and len(changed) == len(clause)
    original, result = _reread(clause, block, text=changed)
    assert result == original
    # A line moved: one more blank line before the last bullet.
    moved = clause[: clause.rindex("●")] + "\n" + clause[clause.rindex("●") :]
    original, result = _reread(clause, block, text=moved)
    assert result == original


@pytest.mark.parametrize(
    "clause",
    [
        LIFELINE,
        SHARED,
        HEATING,
        WOOD,
        "The unit is eligible when any of the following apply:\n\n"
        "● The parent is employed\n\n● The child is disabled",
        "The unit is eligible when one or more of the following apply:\n\n"
        "● The parent is employed\n\n● The child is disabled",
    ],
)
def test_without_an_entry_every_bullet_is_a_gate_as_on_main(clause):
    assert len(_source_conjunctive_fact_gates(clause)) >= 2
