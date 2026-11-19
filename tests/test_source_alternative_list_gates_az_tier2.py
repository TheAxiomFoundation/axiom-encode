"""Reviewed alternative lists of three more Arizona units (shelter, utility, care).

The fixtures are real rejected candidates of one local batch; see
``tests/fixtures/az-tier2-alternative-lists/README.md``. Each unit has a
registry entry bound to its citation and exact text; see
``docs/reviewed-alternative-lists.md``. An entry switches the gates-collapsed
check off for its clause. The tests below pin the effect, the bounds and the
cost of these three entries. They hold while an entry waits for its owner
review and after an owner signs it.
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
    analyze_complete_source_unit,
)

FIXTURES = Path(__file__).parent / "fixtures" / "az-tier2-alternative-lists"
MANUAL = "us-az/manual/des/faa5"
SHELTER = f"{MANUAL}/shelter-expenses-and-deduction/block-16"
UTILITY = f"{MANUAL}/na-utility-expenses-and-allowances/block-17"
CARE = f"{MANUAL}/dependent-care-expense/block-6"

# A synthetic conjunctive list; no entry records it.
CONJUNCTIVE = (
    "The unit is eligible when all of the following apply:\n\n"
    "● The parent is employed\n\n● The child is disabled"
)

SIX = [
    ("shelter-block-16-2eaed082", SHELTER),
    ("shelter-block-16-8176fca8", SHELTER),
    ("utility-block-17-27c00482", UTILITY),
    ("utility-block-17-3f95091c", UTILITY),
    ("dependent-care-block-6-22c86176", CARE),
    ("dependent-care-block-6-deac6f76", CARE),
]
UNITS = SIX[::2]


def _source(directory: str) -> str:
    return (FIXTURES / directory / "source.txt").read_text()


def _collapsed(
    directory: str,
    citation: str,
    content: str | None = None,
    source: str | None = None,
):
    folder = FIXTURES / directory
    profile = vp._numeric_profile_for_citation_path(citation)
    analysis = analyze_complete_source_unit(
        content or (folder / "candidate.yaml").read_text(),
        source or _source(directory),
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


def _without_entries(monkeypatch) -> None:
    monkeypatch.setattr(
        "axiom_encode.harness.source_completeness.reviewed_alternative_lists_for",
        lambda citation, text: (),
    )


@pytest.mark.parametrize(("directory", "citation"), SIX)
def test_candidates_that_encode_alternatives_as_alternatives_pass(directory, citation):
    assert _collapsed(directory, citation) == []


@pytest.mark.parametrize(("directory", "citation"), SIX)
def test_a_candidate_fails_without_its_registry_entry(directory, citation, monkeypatch):
    _without_entries(monkeypatch)
    (issue,) = _collapsed(directory, citation)
    # The finding of the recorded run: the two bullets with a verb as two
    # simultaneous facts, and in the care unit the next paragraph as a third.
    gates = 3 if citation == CARE else 2
    assert issue.expected == f"{gates} distinct gate inputs"
    assert issue.found.startswith(f"0 of {gates} source gates")


@pytest.mark.parametrize(("directory", "citation"), SIX)
@pytest.mark.parametrize("error", [ValueError("malformed"), OSError("unreadable")])
def test_a_candidate_fails_when_the_registry_cannot_be_read(
    directory, citation, error, monkeypatch
):
    def broken():
        raise error

    monkeypatch.setattr(registry, "load_reviewed_alternative_lists", broken)
    assert len(_collapsed(directory, citation)) == 1


@pytest.mark.parametrize(("directory", "citation"), SIX)
def test_an_entry_does_not_apply_under_another_citation(
    directory, citation, monkeypatch
):
    source = _source(directory)
    (entry,) = registry.reviewed_alternative_lists_for(citation, source)
    others = [
        other
        for other in (SHELTER, UTILITY, CARE, citation + "/1", citation.upper(), "")
        if other != citation
    ]
    for other in (*others, None):
        assert registry.reviewed_alternative_lists_for(other, source) == ()
    # The same reviewed text, recorded for another unit, does not help this one.
    elsewhere = dataclasses.replace(entry, citation_path=citation + "9")
    monkeypatch.setattr(
        registry, "load_reviewed_alternative_lists", lambda: (elsewhere,)
    )
    assert registry.reviewed_alternative_lists_for(elsewhere.citation_path, source)
    assert len(_collapsed(directory, citation)) == 1


@pytest.mark.parametrize(("directory", "citation"), SIX)
def test_an_entry_does_not_apply_to_a_changed_text(directory, citation):
    source = _source(directory)
    (entry,) = registry.reviewed_alternative_lists_for(citation, source)
    bullet = entry.lines[1]
    last = bullet.physical_end - 1
    assert source[last] == "n"
    changes = [
        # One character of a recorded bullet.
        source[:last] + "m" + source[last + 1 :],
        # A new bullet in the list.
        source[: bullet.physical_end]
        + "\n\n● The parent is employed"
        + source[bullet.physical_end :],
        # The last character of the unit.
        source[:-1] + "!",
    ]
    # In the care unit the last character is outside the recorded lines: only
    # the digest differs. In the other two units the list ends the unit.
    assert (entry.lines[-1].physical_end < len(source)) is (citation == CARE)
    # Leading or trailing whitespace of the unit is not part of the scanned text.
    for same in (source + "\n", "\n " + source + " \n\n"):
        assert registry.scanned_source_text(same) == source
        assert _collapsed(directory, citation, source=same) == []
    for changed in changes:
        assert changed != source
        assert registry.reviewed_alternative_lists_for(citation, changed) == ()
        assert len(_collapsed(directory, citation, source=changed)) == 1, changed


def _input(payload: dict, name: str) -> None:
    template = payload["inputs"][0]
    payload["inputs"].append(
        {
            "name": name,
            "entity": template["entity"],
            "dtype": "Boolean",
            "period": template["period"],
        }
    )


def _one_rule(payload: dict, fragment: str) -> dict:
    (rule,) = [rule for rule in payload["rules"] if fragment in rule["name"]]
    return rule


def _opaque_occasions(payload: dict) -> None:
    """The four verification occasions as ONE input."""

    _input(payload, "verification_occasion_occurs")
    rule = _one_rule(payload, "verification_required")
    assert " or " in " ".join(rule["versions"][0]["formula"].split())
    rule["versions"][0]["formula"] = "verification_occasion_occurs"


def _occasion_dropped(payload: dict) -> None:
    """One listed occasion is dropped; the others stay."""

    rule = _one_rule(payload, "verification_required")
    terms = " ".join(rule["versions"][0]["formula"].split()).split(" or ")
    assert len(terms) >= 4
    rule["versions"][0]["formula"] = " or ".join(terms[:-1])


def _opaque_statement(payload: dict) -> None:
    """Care unit: every fact of the statement paragraph as ONE input."""

    _input(payload, "adult_care_statement_condition_met")
    rule = _one_rule(payload, "statement_requirement_satisfied")
    rule["versions"][0]["formula"] = "adult_care_statement_condition_met"


@pytest.mark.parametrize(
    ("directory", "citation", "collapse"),
    [
        *((directory, citation, _opaque_occasions) for directory, citation in SIX),
        *((directory, citation, _occasion_dropped) for directory, citation in SIX),
        ("dependent-care-block-6-22c86176", CARE, _opaque_statement),
        ("dependent-care-block-6-deac6f76", CARE, _opaque_statement),
    ],
)
def test_an_entry_switches_the_check_off_for_its_clause(
    directory, citation, collapse, monkeypatch
):
    """Pin the documented cost of these three reviewed exceptions.

    For the recorded clause the check verifies nothing. A candidate that
    merges the four occasions into one input passes this check, and so does a
    candidate that drops an occasion. In the care unit the paragraph about the
    physician or psychologist statement is part of the recorded clause, so a
    candidate that merges its facts into one input passes too. Without the
    entry each of these candidates is refused, as the faithful candidates
    are. The entry's ``not_examined`` and ``owner_must_verify`` fields say so;
    the owner's review of the candidate is the control. If this test fails
    because the check examines the clause again, update the registry fields
    and the docs.
    """

    payload = yaml.safe_load((FIXTURES / directory / "candidate.yaml").read_text())
    collapse(payload)
    content = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    assert _collapsed(directory, citation, content) == []

    _without_entries(monkeypatch)
    assert len(_collapsed(directory, citation, content)) == 1


@pytest.mark.parametrize(("directory", "citation"), UNITS)
def test_the_entry_records_the_whole_clause_and_nothing_after_it(directory, citation):
    source = _source(directory)
    (entry,) = registry.reviewed_alternative_lists_for(citation, source)
    readings = [line.reading for line in entry.lines]
    assert entry.chapeau.physical_start == 0
    assert " when any of the following occur" in entry.chapeau.line
    assert all(line.line.startswith("● ") for line in entry.lines[:4])
    if citation == CARE:
        # The clause runs on into the next paragraph; the rest of the unit is
        # not recorded.
        assert readings == ["alternative"] * 4 + ["outside_condition"]
        assert not entry.lines[-1].line.startswith("●")
        assert entry.lines[-1].line.endswith(".")
        assert entry.lines[-1].physical_end < len(source)
    else:
        # The list is the whole unit.
        assert readings == ["alternative"] * 4
        assert entry.lines[-1].physical_end == len(source)
    assert entry.not_examined and entry.owner_must_verify


@pytest.mark.parametrize(("directory", "citation"), UNITS)
def test_only_the_recorded_clause_loses_its_gates(directory, citation):
    source = _source(directory)
    entries = registry.reviewed_alternative_lists_for(citation, source)
    (entry,) = entries
    end = entry.lines[-1].physical_end
    clause = source[:end]

    def gates(text: str, start: int | None):
        return _source_conjunctive_fact_gates(
            text, reviewed_alternative_lists=entries, text_start=start
        )

    plain = _source_conjunctive_fact_gates(clause)
    assert len(plain) == (3 if citation == CARE else 2)
    assert gates(clause, 0) == ()
    # The clause at another offset, or at no known offset, keeps its gates.
    assert gates(clause, 1) == plain
    assert gates(clause, None) == plain
    # A clause with other bounds keeps its gates: without its last recorded
    # line, and with one more line after it. Without the last line, the care
    # clause is the list alone and still states two facts for this check; the
    # other two clauses then state one, which is no finding with or without
    # an entry.
    shorter = source[: entry.lines[-2].physical_end]
    assert len(_source_conjunctive_fact_gates(shorter)) == (
        2 if citation == CARE else 0
    )
    assert gates(shorter, 0) == _source_conjunctive_fact_gates(shorter)
    longer = clause + "\n\nThe parent is employed and the child is disabled."
    assert len(_source_conjunctive_fact_gates(longer)) > len(plain)
    assert gates(longer, 0) == _source_conjunctive_fact_gates(longer)
    # Another clause is read as without the entry, at any offset.
    conjunctive = _source_conjunctive_fact_gates(CONJUNCTIVE)
    assert len(conjunctive) == 2
    for start in (0, end + 2, None):
        assert gates(CONJUNCTIVE, start) == conjunctive
    # The paragraphs of the unit after the recorded clause are read as without
    # the entry. None of them states two facts for this check, so on the real
    # text this comparison cannot show a finding; the block-4 test in
    # ``tests/test_source_alternative_list_gates.py`` shows one.
    offset = end
    for paragraph in source[end:].split("\n\n"):
        start = source.index(paragraph, offset) if paragraph else offset
        offset = start + len(paragraph)
        if paragraph.strip():
            assert gates(paragraph, start) == _source_conjunctive_fact_gates(paragraph)
            assert len(_source_conjunctive_fact_gates(paragraph)) < 2
