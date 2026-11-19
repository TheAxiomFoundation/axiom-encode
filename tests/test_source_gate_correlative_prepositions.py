"""``both to and from`` joins two prepositions of one fact, not two facts.

The gates-collapsed check splits a source condition on ``and``. In "to
transport the child both to and from a care provider" that split made the tail
"from a care provider" a second simultaneous fact. The correlative ``both``
needs two parallel parts, so its ``and`` cannot start a second clause. Only
that exact wording is kept together; every other ``and`` splits as before. All
text here is synthetic.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _source_conjunctive_fact_gates,
    _source_gate_split_conjunctive_conditions,
    analyze_complete_source_unit,
)
from tests import test_source_credit_difference_conditions as credit

CITATION = "us-zz/manual/synthetic/travel-allowance/block-1"
ROUND_TRIP = (
    "The travel allowance is paid when a member incurs a cost to drive their "
    "dependent to, from, or both to and from a dependent care provider."
)
EXCEPTION = (
    "However, the travel allowance is not paid when the member resides with "
    "the dependent care provider."
)


def _gate_count(text: str) -> int:
    return len(_source_conjunctive_fact_gates(text))


def _candidate(condition_excerpt: str, formula: str, inputs: list[str]) -> str:
    def atom(kind: str, excerpt: str) -> dict:
        return {
            "path": "versions[0].formula",
            "kind": kind,
            "source": {"corpus_citation_path": CITATION, "excerpt": excerpt},
        }

    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "proof_validation": {"required": True},
                "source_verification": {"corpus_citation_path": CITATION},
                "summary": "Synthetic travel allowance.",
            },
            "inputs": [
                {
                    "name": name,
                    "entity": "Household",
                    "dtype": "Boolean",
                    "period": "Month",
                }
                for name in inputs
            ],
            "rules": [
                {
                    "name": "travel_allowance_is_paid",
                    "kind": "derived",
                    "entity": "Household",
                    "dtype": "Judgment",
                    "period": "Month",
                    "source": "Synthetic travel allowance, block 1",
                    "metadata": {
                        "proof": {
                            "atoms": [
                                atom("condition", condition_excerpt),
                                atom("exception", EXCEPTION),
                            ]
                        }
                    },
                    "versions": [{"effective_from": "2026-01-01", "formula": formula}],
                }
            ],
        },
        sort_keys=False,
    )


def _collapsed(source: str, content: str) -> list:
    profile = vp._numeric_profile_for_citation_path(CITATION)
    analysis = analyze_complete_source_unit(
        content,
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


ONE_FACT_FORMULA = (
    "if member_resides_with_dependent_care_provider: false\n"
    "else: member_incurs_cost_driving_dependent_to_or_from_care_provider"
)
ONE_FACT_INPUTS = [
    "member_resides_with_dependent_care_provider",
    "member_incurs_cost_driving_dependent_to_or_from_care_provider",
]


def test_round_trip_wording_is_one_fact():
    assert _gate_count(ROUND_TRIP) == 0


def test_round_trip_candidate_with_one_input_is_not_refused():
    source = ROUND_TRIP + "\n\n" + EXCEPTION
    content = _candidate(ROUND_TRIP, ONE_FACT_FORMULA, ONE_FACT_INPUTS)
    assert _collapsed(source, content) == []


@pytest.mark.parametrize(
    "condition",
    [
        # The plain phrase has no correlative: it stays split.
        ROUND_TRIP.replace("to, from, or both to and from", "to and from"),
        # Two separate journeys are two facts.
        ROUND_TRIP.replace(
            "to, from, or both to and from a dependent care provider",
            "to a dependent care provider and to drive their dependent from "
            "a dependent care school",
        ),
        # Another object between the prepositions.
        ROUND_TRIP.replace("both to and from", "both to work and from"),
        # Other correlatives and connectors.
        ROUND_TRIP.replace("both to and from", "both to and away from"),
        ROUND_TRIP.replace("both to and from", "either to and from"),
        ROUND_TRIP.replace("both to and from", "them to and from"),
        # A longer word that only ends in "both" or starts with "from".
        ROUND_TRIP.replace("both to and from", "saboth to and from"),
        ROUND_TRIP.replace("both to and from", "both to and fromward"),
        ROUND_TRIP.replace("both to and from", "both to nor from"),
        # Changed spacing is not the reviewed wording.
        ROUND_TRIP.replace("both to and from", "both to  and from"),
        ROUND_TRIP.replace("both to and from", "both to and\nfrom"),
    ],
)
def test_other_wordings_stay_two_facts_and_stay_refused(condition):
    assert _gate_count(condition) == 2
    source = condition + "\n\n" + EXCEPTION
    content = _candidate(condition, ONE_FACT_FORMULA, ONE_FACT_INPUTS)
    assert len(_collapsed(source, content)) == 1


def test_later_conjunction_after_the_phrase_still_splits():
    text = ROUND_TRIP.replace(
        "provider.", "provider and the member is employed and the member is insured."
    )
    without_phrase = text.replace("to, from, or both to and from", "to")
    assert _gate_count(text) == _gate_count(without_phrase) == 3
    source = text + "\n\n" + EXCEPTION
    content = _candidate(text, ONE_FACT_FORMULA, ONE_FACT_INPUTS)
    assert len(_collapsed(source, content)) == 1


@pytest.mark.parametrize(
    "tail",
    [
        "a dependent care provider.",
        "a dependent care provider who is insured.",
        "work and the member is employed.",
        "a dependent care provider and the dependent is disabled and the "
        "member is a resident.",
        "school, if the member is employed and the dependent is blind.",
    ],
)
def test_phrase_reads_exactly_like_a_single_preposition(tail):
    """The kept phrase adds no fact and hides none: it counts like ``to``."""
    head = "The allowance is paid when a member drives their dependent "
    assert _gate_count(head + "both to and from " + tail) == _gate_count(
        head + "to " + tail
    )


def test_split_is_unchanged_without_the_keyword():
    body = " a member drives their dependent both to and from a provider."
    assert len(_source_gate_split_conjunctive_conditions(body)) == 2
    kept = _source_gate_split_conjunctive_conditions(
        body, keep_correlative_prepositions=True
    )
    assert kept == [(None, body)]


def test_only_the_correlative_and_is_kept():
    body = " both to and from a provider and both to and from a school and to and from work"
    kept = _source_gate_split_conjunctive_conditions(
        body, keep_correlative_prepositions=True
    )
    assert [segment for _connector, segment in kept] == [
        " both to and from a provider ",
        " both to and from a school ",
        " to ",
        " from work",
    ]
    assert [connector for connector, _segment in kept] == [None, "and", "and", "and"]


def test_case_is_not_significant():
    assert _gate_count(ROUND_TRIP.replace("both to and from", "Both To And From")) == 0


def test_phrase_on_a_later_line_is_kept_at_its_own_offset():
    """Offsets are per physical line: the phrase on line two is still found."""
    body = (
        " the member is employed and insured\n"
        " the member drives both to and from a provider and is a resident"
    )
    kept = _source_gate_split_conjunctive_conditions(
        body, keep_correlative_prepositions=True
    )
    assert kept == [
        (None, " the member is employed "),
        ("and", " insured"),
        (None, " the member drives both to and from a provider "),
        ("and", " is a resident"),
    ]


def test_phrase_after_a_yet_boundary_is_kept():
    """A ``yet`` boundary becomes a line break before the split."""
    body = (
        " the member is employed, yet the member drives both to and from a "
        "provider and is a resident"
    )
    kept = _source_gate_split_conjunctive_conditions(
        body, keep_correlative_prepositions=True
    )
    assert len(kept) == 3
    assert [connector for connector, _segment in kept] == [None, None, "and"]
    assert "both to and from a provider" in kept[1][1]
    # The same text with the phrase on the first line, for the offsets.
    shifted = _source_gate_split_conjunctive_conditions(
        " the member drives both to and from a provider, yet the member is "
        "employed and is a resident",
        keep_correlative_prepositions=True,
    )
    assert len(shifted) == 3
    assert "both to and from a provider" in shifted[0][1]


@pytest.mark.parametrize(
    "phrase",
    [
        "both to and  from",
        "both to  and from",
        "both  to and from",
        "both to and\tfrom",
        "both to\tand from",
        "both\tto and from",
        "both to and\u00a0from",
        "both to\u00a0and from",
        "both\u00a0to and from",
        # Not the word "both".
        "_both to and from",
        "2both to and from",
    ],
)
def test_other_spacing_and_other_tokens_stay_split(phrase):
    body = f" a member drives their dependent {phrase} a provider."
    kept = _source_gate_split_conjunctive_conditions(
        body, keep_correlative_prepositions=True
    )
    assert len(kept) == 2
    assert kept == _source_gate_split_conjunctive_conditions(body)


def test_credit_difference_reader_still_splits_the_phrase(monkeypatch):
    """The other caller of the split keeps the default: the phrase splits."""
    calls = []
    split = completeness._source_gate_split_conjunctive_conditions

    def recording_split(text, **keywords):
        segments = split(text, **keywords)
        calls.append((text, keywords, segments))
        return segments

    monkeypatch.setattr(
        completeness, "_source_gate_split_conjunctive_conditions", recording_split
    )
    clause = credit.CLAUSE.replace(
        "has a federal tax liability", "is driven both to and from work"
    )
    payload, cases = credit._credit_candidate(source=clause)
    credit._public_gate(payload, clause, cases)
    with_phrase = [call for call in calls if "both to and from" in call[0]]
    by_default = [call for call in with_phrase if not call[1]]
    assert by_default, "the credit-difference reader was not reached"
    assert all(len(segments) == 2 for _text, _keywords, segments in by_default)
    assert all(
        keywords == {"keep_correlative_prepositions": True} and len(segments) == 1
        for _text, keywords, segments in with_phrase
        if keywords
    )
